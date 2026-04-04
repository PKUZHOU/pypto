# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO backend driver.

Orchestrates the full PTO backend output pipeline:

- **Kernel files**: InCore functions go through C++ PTOCodegen (IR → MLIR) → ptoas → kernel wrapper
- **Orchestration**: Shared C++ orchestration codegen (PTO2 runtime API)
- **Config**: Generates kernel_config.py with runtime/orchestration/kernel metadata

Entry point: ``generate(program, output_dir) -> dict[str, str]``
"""

import logging
import os
import pprint
import re
import shutil
import subprocess
import textwrap
from collections import OrderedDict

from pypto.pypto_core import backend as _backend_core
from pypto.pypto_core import codegen as _codegen_core
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core import DataType as _DataType
from pypto.language.distributed import (
    DistributedCompileContext,
    DistributedAllReducePhase,
    DistributedLocalPhase,
    DistributedPhase,
    DistributedProgram,
    PhaseArg,
    call_source_fn,
    lookup_distributed_program,
)

logger = logging.getLogger(__name__)

_PTOAS_RELEASE_URL = "https://github.com/zhangstevenunity/PTOAS/releases"


class PartialCodegenError(RuntimeError):
    """Codegen failed after producing some output files."""

    def __init__(self, message: str, files: dict[str, str]) -> None:
        super().__init__(message)
        self.files = files


_ERROR_LINE_RE = re.compile(r"(?:^Error:|\berror:)", re.IGNORECASE)


def _strip_function_name_prefix(summary: str, func_name: str) -> str:
    """Remove a leading function-name prefix without corrupting file paths."""
    prefixes = (
        f"Failed to compile function '{func_name}':",
        f'Failed to compile function "{func_name}":',
        f"{func_name}:",
        func_name,
    )
    for prefix in prefixes:
        if summary.startswith(prefix):
            stripped = summary[len(prefix) :].strip()
            if stripped:
                return stripped
    return summary


def _get_error_summary(exc: Exception, func_name: str) -> str:
    """Extract the most useful error line from an exception.

    Strips the C++ Traceback tail, prefers the first line that contains an
    actual error marker, and removes only a leading *func_name* prefix so file
    paths remain intact.
    """
    msg = str(exc)
    traceback_marker = msg.find("\n\nC++ Traceback")
    if traceback_marker != -1:
        msg = msg[:traceback_marker]
    lines = [line.strip() for line in msg.splitlines() if line.strip()]
    if not lines:
        return type(exc).__name__

    first_line = lines[0]
    ptoas_prefix = "ptoas compilation failed:"
    if first_line.startswith(ptoas_prefix):
        first_detail = first_line[len(ptoas_prefix) :].strip()
        detail_lines = []
        if first_detail:
            detail_lines.append(first_detail)
        detail_lines.extend(lines[1:])
        for line in detail_lines:
            if _ERROR_LINE_RE.search(line):
                return f"{ptoas_prefix} {line}"

    for line in lines:
        if _ERROR_LINE_RE.search(line):
            return _strip_function_name_prefix(line, func_name)

    return _strip_function_name_prefix(first_line, func_name)


def _format_error_report(
    errors: list[tuple[str, Exception]],
    output_dir: str,
) -> str:
    """Build a concise error summary table and write full details to a log file.

    Each failed function is shown on its own row, with ``Function`` as the
    first column and ``Error`` as the second column. Returns the summary string
    for use in the ``RuntimeError`` message.
    """
    max_error_col = 60

    summaries = OrderedDict((name, _get_error_summary(exc, name)) for name, exc in errors)
    longest_error = max(len(summary) for summary in summaries.values())
    error_col_width = min(longest_error, max_error_col) + 2
    error_col_width = max(error_col_width, len("Error") + 2)
    func_col_width = max(len(n) for n, _ in errors) + 2
    func_col_width = max(func_col_width, len("Function") + 2)

    lines: list[str] = [f"{len(errors)} function(s) failed to compile:\n"]
    lines.append(f"  {'Function':<{func_col_width}}| {'Error'}")
    lines.append(f"  {'-' * func_col_width}+{'-' * error_col_width}")

    sep_line = f"  {'-' * func_col_width}+{'-' * error_col_width}"
    for func_name, summary in summaries.items():
        wrapped = textwrap.wrap(summary, width=max_error_col) or [summary]
        lines.append(f"  {func_name:<{func_col_width}}| {wrapped[0]}")
        for err_part in wrapped[1:]:
            lines.append(f"  {'':<{func_col_width}}| {err_part}")
        lines.append(sep_line)

    summary_text = "\n".join(lines)

    report_dir = os.path.join(output_dir, "report")
    detail_path = os.path.join(report_dir, "codegen_errors.txt")
    separator = "\n" + "=" * 72 + "\n"
    detail_parts = [f"  [{name}]\n{exc}" for name, exc in errors]
    detail_content = summary_text + "\n\n" + separator.join(detail_parts)
    try:
        os.makedirs(report_dir, exist_ok=True)
        with open(detail_path, "w") as f:
            f.write(detail_content)
        lines.append(f"\n  Full details: {detail_path}")
    except OSError:
        pass

    return "\n".join(lines)


def _run_ptoas(
    pto_path: str,
    output_path: str,
    ptoas_flags: list[str] | None = None,
) -> None:
    """Run the ptoas tool to compile a .pto file to C++.

    Locates ptoas via PTOAS_ROOT env var (``$PTOAS_ROOT/ptoas``) or PATH fallback.

    Args:
        pto_path: Path to the input .pto file
        output_path: Path for the output .cpp file
        ptoas_flags: Additional flags to pass to ptoas (optional)

    Raises:
        FileNotFoundError: If the ptoas binary cannot be found
        RuntimeError: If ptoas compilation fails
    """
    ptoas_root = os.environ.get("PTOAS_ROOT")
    if ptoas_root:
        ptoas_bin = os.path.join(ptoas_root, "ptoas")
        if not (os.path.isfile(ptoas_bin) and os.access(ptoas_bin, os.X_OK)):
            raise FileNotFoundError(
                f"PTOAS_ROOT is set to '{ptoas_root}' but '{ptoas_bin}' does not exist or is not executable. "
            )
    else:
        ptoas_bin = shutil.which("ptoas")
        if not ptoas_bin:
            raise FileNotFoundError(
                "ptoas binary not found. Set PTOAS_ROOT to the extracted release directory, "
                f"or add ptoas to your PATH.\nDownload from: {_PTOAS_RELEASE_URL}"
            )

    cmd = [ptoas_bin, pto_path, "-o", output_path]
    if ptoas_flags:
        cmd.extend(ptoas_flags)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ptoas compilation timed out after {exc.timeout}s") from exc
    if result.returncode != 0:
        raise RuntimeError(f"ptoas compilation failed: {result.stderr.strip()}")


_KERNEL_HEADER = """\
// Kernel Function: {func_name}
// Generated by PyPTO IR Compiler (PTO backend)

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

"""


def _preprocess_ptoas_output(content: str) -> str:
    """Strip includes/using and make functions static in ptoas output.

    Removes the header lines that the wrapper already provides, and replaces
    ``__global__ AICORE void`` with ``static __aicore__ void`` so the wrapper's
    ``kernel_entry`` is the actual entry point.
    """
    lines = content.splitlines(keepends=True)
    filtered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#include") and (
            "pto-inst" in stripped or "cstdint" in stripped or "tensor.h" in stripped
        ):
            continue
        if stripped == "using namespace pto;":
            continue
        filtered.append(line)
    result = "".join(filtered)
    # Non-mixed kernels: __global__ AICORE void → static __aicore__ void
    result = re.sub(r"__global__\s+AICORE\s+void", "static __aicore__ void", result)
    # Mixed kernel sub-functions: bare AICORE void at line start → static __aicore__ void
    result = re.sub(r"^\s*AICORE(?=\s+void)", "static __aicore__", result, flags=re.MULTILINE)
    # Helper functions: static AICORE inline → static __aicore__ inline
    result = re.sub(r"\bAICORE\b", "__aicore__", result)
    return result


def _generate_arg_unpacking(func: _ir_core.Function) -> tuple[str, list[str]]:
    """Generate C++ code to unpack ``int64_t* args`` into typed locals.

    Args[] are dispatched in tensors-first order (all tensors, then scalars),
    matching the PTOParam dispatch convention and the MLIR func.func signature
    order emitted by PTOCodegen. The returned var_names list is also in
    tensors-first order, matching the compiled ptoas function parameter order.

    Returns:
        A tuple of (C++ unpacking code, list of local variable names in
        tensors-first order).
    """
    lines: list[str] = []
    var_names: list[str] = []

    # Separate params into tensors and scalars for tensors-first dispatch order
    tensor_params = [p for p in func.params if isinstance(p.type, _ir_core.TensorType)]
    scalar_params = [p for p in func.params if isinstance(p.type, _ir_core.ScalarType)]
    other_params = [
        p for p in func.params if not isinstance(p.type, (_ir_core.TensorType, _ir_core.ScalarType))
    ]
    if other_params:
        raise ValueError(
            f"Unsupported parameter type(s) for wrapper generation in function {func.name}: "
            + ", ".join(f"{p.name_hint}: {type(p.type).__name__}" for p in other_params)
        )

    scalar_start_idx = len(tensor_params)

    # Unpack tensors: args[0..N_tensors-1]
    for i, param in enumerate(tensor_params):
        param_name = param.name_hint
        assert isinstance(param.type, _ir_core.TensorType)
        c_type = param.type.dtype.to_c_type_string()
        lines.append(f"    // Unpack tensor: {param_name}")
        lines.append(f"    __gm__ Tensor* {param_name}_tensor = reinterpret_cast<__gm__ Tensor*>(args[{i}]);")
        lines.append(
            f"    __gm__ {c_type}* {param_name} = "
            f"reinterpret_cast<__gm__ {c_type}*>("
            f"{param_name}_tensor->buffer.addr) + {param_name}_tensor->start_offset;"
        )
        lines.append("")
        var_names.append(param_name)

    # Unpack scalars: args[N_tensors..]
    for j, param in enumerate(scalar_params):
        param_name = param.name_hint
        assert isinstance(param.type, _ir_core.ScalarType)
        c_type = param.type.dtype.to_c_type_string()
        arg_idx = scalar_start_idx + j
        lines.append(f"    // Unpack scalar: {param_name}")
        lines.append(f"    union {{ uint64_t u64; {c_type} val; }} {param_name}_conv;")
        lines.append(f"    {param_name}_conv.u64 = args[{arg_idx}];")
        lines.append(f"    {c_type} {param_name} = {param_name}_conv.val;")
        lines.append("")
        var_names.append(param_name)

    # Extract dynamic dimension values from tensor structs (shapes[] holds current view shape at runtime)
    seen_dyn_vars: set[str] = set()
    for param in tensor_params:
        assert isinstance(param.type, _ir_core.TensorType)
        for dim_idx, dim in enumerate(param.type.shape):
            if isinstance(dim, _ir_core.Var) and dim.name_hint not in seen_dyn_vars:
                var_name = dim.name_hint
                seen_dyn_vars.add(var_name)
                lines.append(f"    // Extract dynamic dim: {var_name}")
                lines.append(
                    f"    int64_t {var_name} = static_cast<int64_t>"
                    f"({param.name_hint}_tensor->shapes[{dim_idx}]);"
                )
                lines.append("")
                var_names.append(var_name)

    return "\n".join(lines), var_names


def _generate_kernel_wrapper(func: _ir_core.Function, ptoas_code: str) -> str:
    """Generate a complete kernel wrapper file for one InCore function.

    Combines:
    1. Kernel header (includes, macros)
    2. Preprocessed ptoas code (static, no duplicate includes)
    3. ``kernel_entry`` wrapper with arg unpacking and forward call
    """
    header = _KERNEL_HEADER.format(func_name=func.name)
    ptoas_body = _preprocess_ptoas_output(ptoas_code)
    unpacking_code, var_names = _generate_arg_unpacking(func)
    call_args = ", ".join(var_names)

    wrapper_func = (
        "// --- Kernel entry point ---\n"
        'extern "C" __aicore__ __attribute__((always_inline)) '
        "void kernel_entry(__gm__ int64_t* args)\n"
        "{\n"
        f"{unpacking_code}\n"
        f"    // Forward to ptoas-generated function\n"
        f"    {func.name}({call_args});\n"
        "}\n"
    )

    return f"{header}\n// --- ptoas-generated code ---\n{ptoas_body}\n{wrapper_func}"


def _serialize_distributed_spec(spec: dict[str, object]) -> str:
    """Return a stable Python literal for ``DISTRIBUTED_SPEC``."""

    return pprint.pformat(spec, width=100, sort_dicts=False)


def _generate_config_file(
    *,
    orch_source_name: str,
    orch_function_name: str,
    kernel_entries: list[dict[str, object]],
    runtime_config: dict[str, object] | None = None,
    runtime_env: dict[str, str] | None = None,
    distributed_spec: dict[str, object] | None = None,
) -> str:
    """Generate kernel_config.py content."""
    runtime_config = runtime_config or {
        "runtime": "tensormap_and_ringbuffer",
        "aicpu_thread_num": 4,
        "orch_thread_num": 1,
        "block_dim": 24,
    }
    lines = [
        "# Kernel and Orchestration Configuration\n",
        "from pathlib import Path\n",
        "_ROOT_DIR = Path(__file__).parent\n",
        f"RUNTIME_CONFIG = {pprint.pformat(runtime_config, width=100, sort_dicts=False)}\n",
    ]
    if runtime_env:
        lines.append(f"RUNTIME_ENV = {pprint.pformat(runtime_env, width=100, sort_dicts=False)}\n")
    lines += [
        "ORCHESTRATION = {",
        f'\t"source": str(_ROOT_DIR / "orchestration" / "{orch_source_name}.cpp"),',
        f'\t"function_name": "{orch_function_name}"',
        "}\n",
        "KERNELS = [",
    ]

    for entry in sorted(kernel_entries, key=lambda item: int(item["func_id"])):
        name = str(entry["name"])
        func_id = int(entry["func_id"])
        ct_str = str(entry["core_type"])
        lines.append(
            f'\t{{"func_id": {func_id}, '
            f'"source": str(_ROOT_DIR / "kernels" / "{ct_str}" / "{name}.cpp"), '
            f'"core_type": "{ct_str}"}},'
        )

    lines.append("]")
    if distributed_spec is not None:
        lines.append("")
        lines.append(f"DISTRIBUTED_SPEC = {_serialize_distributed_spec(distributed_spec)}")
    return "\n".join(lines) + "\n"


def _extract_group_member_names(
    group_func: _ir_core.Function,
) -> list[str]:
    """Extract function names called by a Group function from its body."""
    names: list[str] = []
    stmts = _ir_core.flatten_to_stmts(group_func.body)
    for stmt in stmts:
        call = None
        if isinstance(stmt, _ir_core.EvalStmt):
            call = stmt.expr
        elif isinstance(stmt, _ir_core.AssignStmt):
            call = stmt.value
        if isinstance(call, _ir_core.Call) and isinstance(call.op, _ir_core.GlobalVar):
            names.append(call.op.name)
    return names


def _build_group_mapping(
    program: _ir_core.Program,
) -> tuple[dict[str, list[_ir_core.Function]], list[_ir_core.Function]]:
    """Partition InCore functions into groups and ungrouped.

    Returns:
        (groups, ungrouped) where groups maps group_name to list of member
        InCore functions, and ungrouped is a list of InCore functions not
        belonging to any group.
    """
    func_by_name: dict[str, _ir_core.Function] = {f.name: f for f in program.functions.values()}
    grouped_names: set[str] = set()
    groups: dict[str, list[_ir_core.Function]] = {}

    for func in program.functions.values():
        if func.func_type != _ir_core.FunctionType.Group:
            continue
        member_names = _extract_group_member_names(func)
        members = [func_by_name[n] for n in member_names if n in func_by_name]
        if members:
            groups[func.name] = members
            grouped_names.update(n for n in member_names if n in func_by_name)

    ungrouped = [
        f
        for f in program.functions.values()
        if _ir_core.is_incore_type(f.func_type) and f.name not in grouped_names
    ]
    return groups, ungrouped


def _core_type_to_config_string(core_type: _ir_core.CoreType) -> str:
    return "aiv" if core_type == _ir_core.CoreType.VECTOR else "aic"


def _serialize_phase_args(args: list[PhaseArg] | None) -> list[dict[str, str]] | None:
    if args is None:
        return None
    return [{"token": arg.token, "kind": arg.kind} for arg in args]


def _build_distributed_spec(
    metadata: DistributedProgram,
    phase_specs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    spec: dict[str, object] = {
        "buffer_attrs": {
            attr.name: {
                key: value
                for key, value in {
                    "placement": attr.placement,
                    "data_prefix_elems": int(attr.data_prefix_elems),
                }.items()
                if value is not None and not (key == "data_prefix_elems" and value == 0)
            }
            for attr in metadata.buffer_attrs
        },
        "win_sync_prefix": int(metadata.win_sync_prefix),
    }
    if phase_specs is None:
        phase_specs = []
        for phase in metadata.phases:
            phase_entry = None
            if isinstance(phase, DistributedLocalPhase):
                phase_entry = {
                    "orch_func": phase.orch_func,
                    "barrier_before": bool(phase.barrier_before),
                }
                args = _serialize_phase_args(phase.args)
                if args is not None:
                    phase_entry["args"] = args
            elif isinstance(phase, DistributedAllReducePhase):
                phase_entry = {
                    "orch_func": phase.orch_func,
                    "barrier_before": bool(phase.barrier_before),
                    "args": _serialize_phase_args(
                        [
                            PhaseArg(phase.input_name, "tensor"),
                            PhaseArg(phase.output_name, "tensor"),
                            PhaseArg(phase.nranks_arg, "scalar"),
                            PhaseArg(phase.root_arg, "scalar"),
                            PhaseArg(phase.device_ctx_arg, "scalar"),
                        ]
                    ),
                }
            elif isinstance(phase, DistributedPhase):
                phase_entry = {
                    "orch_func": phase.orch_func,
                    "barrier_before": bool(phase.barrier_before),
                }
                args = _serialize_phase_args(phase.args)
                if args is not None:
                    phase_entry["args"] = args
            if phase_entry is not None:
                phase_specs.append(phase_entry)
    if phase_specs:
        spec["phases"] = phase_specs
    if metadata.args is not None:
        spec["args"] = list(metadata.args)
    if metadata.inputs is not None:
        spec["inputs"] = list(metadata.inputs)
    if metadata.outputs is not None:
        spec["outputs"] = list(metadata.outputs)
    if metadata.comm_include_dirs:
        spec["comm_include_dirs"] = list(metadata.comm_include_dirs)
    return spec


def _resolve_function(program: _ir_core.Program, func_name: str) -> _ir_core.Function:
    for func in program.functions.values():
        if getattr(func, "name", None) == func_name:
            return func
    raise RuntimeError(f"Distributed phase references unknown function '{func_name}'")


def _default_phase_args_from_function(func: _ir_core.Function) -> list[PhaseArg]:
    result: list[PhaseArg] = []
    for param in func.params:
        token = _strip_ssa_name(param.name_hint)
        if isinstance(param.type, _ir_core.TensorType):
            result.append(PhaseArg(token, "tensor"))
        elif isinstance(param.type, _ir_core.ScalarType):
            result.append(PhaseArg(token, "scalar"))
        else:
            raise RuntimeError(
                f"Distributed local phase '{func.name}' uses unsupported param type {type(param.type).__name__}"
            )
    return result


def _strip_ssa_name(name: str) -> str:
    return re.sub(r"__ssa_v\d+$", "", name)


def _resolve_tensor_type_by_name(program: _ir_core.Program, tensor_name: str) -> _ir_core.TensorType:
    for func in program.functions.values():
        for param in func.params:
            if isinstance(param.type, _ir_core.TensorType) and _strip_ssa_name(param.name_hint) == tensor_name:
                return param.type
    raise RuntimeError(f"Distributed phase references unknown tensor '{tensor_name}'")


def _extract_orchestration_entry(code: str, new_name: str) -> str:
    marker = '__attribute__((visibility("default")))\nvoid aicpu_orchestration_entry'
    start = code.find(marker)
    if start < 0:
        raise RuntimeError("Failed to locate aicpu_orchestration_entry in generated orchestration source")
    brace_start = code.find("{", start)
    if brace_start < 0:
        raise RuntimeError("Failed to locate orchestration function body start")
    depth = 0
    end = -1
    for idx in range(brace_start, len(code)):
        ch = code[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end < 0:
        raise RuntimeError("Failed to locate orchestration function body end")
    func_code = code[start:end].replace(
        "void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args, int orch_thread_num, int orch_thread_index)",
        f"void {new_name}(PTO2Runtime* rt, const ChipStorageTaskArgs& orch_args, int orch_thread_num, int orch_thread_index)",
        1,
    )
    replacements = {
        "PTO2_SCOPE() {": "PTO2_SCOPE(rt) {",
        "pto2_rt_submit_aic_task(": "pto2_rt_submit_aic_task(rt, ",
        "pto2_rt_submit_aiv_task(": "pto2_rt_submit_aiv_task(rt, ",
        "pto2_rt_submit_task(": "pto2_rt_submit_task(rt, ",
        "pto2_rt_add_dependency(": "pto2_rt_add_dependency(rt, ",
        "TaskOutputTensors ": "SubmitResult ",
        ".get_ref(": ".outputs.get_ref(",
    }
    for old, new in replacements.items():
        func_code = func_code.replace(old, new)
    return func_code


def _generate_distributed_phase_file(phase_functions: list[str]) -> str:
    parts = [
        "#include <stddef.h>",
        "#include <stdint.h>",
        "#include <stdio.h>",
        "",
        '#include "pto_orchestration_api.h"',
        "",
        'extern "C" {',
        "",
        '__attribute__((visibility("default")))',
        "PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {",
        "    return PTO2OrchestrationConfig{",
        "        .expected_arg_count = orch_args.tensor_count() + orch_args.scalar_count(),",
        "    };",
        "}",
        "",
    ]
    for phase_function in phase_functions:
        parts.append(phase_function.rstrip())
        parts.append("")
    parts.append('}  // extern "C"')
    parts.append("")
    return "\n".join(parts)


def _generate_allreduce_phase_source(
    *,
    orch_func: str,
    kernel_func_id: int,
    shape: list[int],
    dtype: str,
) -> str:
    shape_values = ", ".join(str(dim) for dim in shape)
    return textwrap.dedent(
        f"""\
        __attribute__((visibility("default")))
        void {orch_func}(
            PTO2Runtime* rt, const ChipStorageTaskArgs& orch_args, int orch_thread_num, int orch_thread_index
        ) {{
            (void)orch_thread_num;
            (void)orch_thread_index;

            uint32_t tensor_shapes[{len(shape)}] = {{{shape_values}}};
            Tensor ext_input = make_tensor_external(
                reinterpret_cast<void*>(orch_args.scalar(0)),
                tensor_shapes,
                {len(shape)},
                {dtype}
            );
            Tensor ext_output = make_tensor_external(
                reinterpret_cast<void*>(orch_args.scalar(1)),
                tensor_shapes,
                {len(shape)},
                {dtype}
            );
            uint64_t nranks = orch_args.scalar(2);
            uint64_t root = orch_args.scalar(3);
            uint64_t device_ctx = orch_args.scalar(4);

            PTO2_SCOPE(rt) {{
                Arg allreduce_args;
                allreduce_args.add_input(ext_input);
                allreduce_args.add_inout(ext_output);
                allreduce_args.add_scalar(nranks);
                allreduce_args.add_scalar(root);
                allreduce_args.add_scalar(device_ctx);
                pto2_rt_submit_aiv_task(rt, {kernel_func_id}, allreduce_args);
            }}
        }}
        """
    )


def _generate_allreduce_kernel_source(chunk_elems: int) -> str:
    return textwrap.dedent(
        f"""\
        /**
         * Auto-generated float32 allreduce(sum) kernel for distributed PyPTO phases.
         * args layout:
         *   [0] = input tensor handle   (window-backed Tensor*)
         *   [1] = output tensor handle  (device-backed Tensor*)
         *   [2] = nranks
         *   [3] = root (unused)
         *   [4] = CommDeviceContext*
         */
        #include <cstdint>
        #include <pto/pto-inst.hpp>
        #include "pto/comm/comm_types.hpp"
        #include "common/comm_context.h"
        #include "tensor.h"

        #ifndef __gm__
        #define __gm__
        #endif

        #ifndef __aicore__
        #define __aicore__ [aicore]
        #endif

        static constexpr size_t ALLREDUCE_CHUNK = {chunk_elems};
        static constexpr int kMaxSupportedRanks = 16;

        template <typename T>
        AICORE inline __gm__ T* CommRemotePtr(__gm__ CommDeviceContext* ctx, __gm__ T* localPtr, int pe) {{
            uint64_t localBase = ctx->windowsIn[ctx->rankId];
            uint64_t offset = (uint64_t)localPtr - localBase;
            return (__gm__ T*)(ctx->windowsIn[pe] + offset);
        }}

        extern "C" __aicore__ __attribute__((always_inline))
        void kernel_entry(__gm__ int64_t* args) {{
            __gm__ Tensor* input_tensor = reinterpret_cast<__gm__ Tensor*>(args[0]);
            __gm__ Tensor* output_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
            __gm__ float* input = reinterpret_cast<__gm__ float*>(input_tensor->buffer.addr);
            __gm__ float* output = reinterpret_cast<__gm__ float*>(output_tensor->buffer.addr);
            int nranks = static_cast<int>(args[2]);
            __gm__ CommDeviceContext* commCtx = reinterpret_cast<__gm__ CommDeviceContext*>(args[4]);
            using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
            using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
            using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
            using TileData =
                pto::Tile<pto::TileType::Vec, float, 1, ALLREDUCE_CHUNK, pto::BLayout::RowMajor, -1, -1>;

            if (input_tensor->dtype != DataType::FLOAT32 || output_tensor->dtype != DataType::FLOAT32) {{
                pipe_barrier(PIPE_ALL);
                return;
            }}
            if (nranks <= 0 || nranks > kMaxSupportedRanks) {{
                pipe_barrier(PIPE_ALL);
                return;
            }}

            if (input_tensor->ndims == 0 || output_tensor->ndims != input_tensor->ndims) {{
                pipe_barrier(PIPE_ALL);
                return;
            }}

            uint64_t total_elems = 1;
            for (uint32_t i = 0; i < input_tensor->ndims; ++i) {{
                if (output_tensor->shapes[i] != input_tensor->shapes[i]) {{
                    pipe_barrier(PIPE_ALL);
                    return;
                }}
                total_elems *= static_cast<uint64_t>(input_tensor->shapes[i]);
            }}

            TileData accTile(1, ALLREDUCE_CHUNK);
            TileData recvTile(1, ALLREDUCE_CHUNK);
            TASSIGN(accTile, 0x0);
            TASSIGN(recvTile, 0x10000);

            for (uint64_t offset = 0; offset < total_elems; offset += ALLREDUCE_CHUNK) {{
                uint64_t chunk = total_elems - offset;
                if (chunk > ALLREDUCE_CHUNK) {{
                    chunk = ALLREDUCE_CHUNK;
                }}

                ShapeDyn shape(1, 1, 1, 1, chunk);
                StrideDyn stride(chunk, chunk, chunk, chunk, 1);
                Global outputG(output + offset, shape, stride);

                __gm__ float* firstInput = CommRemotePtr(commCtx, input + offset, 0);
                Global firstG(firstInput, shape, stride);
                TLOAD(accTile, firstG);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

                for (int r = 1; r < nranks; ++r) {{
                    __gm__ float* remoteInput = CommRemotePtr(commCtx, input + offset, r);
                    Global remoteG(remoteInput, shape, stride);
                    TLOAD(recvTile, remoteG);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                    TADD(accTile, accTile, recvTile);
                    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                }}

                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TSTORE(outputG, accTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }}
            pipe_barrier(PIPE_ALL);
        }}
        """
    )


def _const_int_value(expr: object) -> int:
    if isinstance(expr, _ir_core.ConstInt):
        return int(expr.value)
    raise RuntimeError(f"Only static tensor shapes are supported in distributed local phases, got {expr!r}")


def _tensor_type_to_runtime_dtype(dtype: object) -> str:
    mapping = {
        "fp16": "DataType::FLOAT16",
        "fp32": "DataType::FLOAT32",
        "bf16": "DataType::BFLOAT16",
        "int8": "DataType::INT8",
        "uint8": "DataType::UINT8",
        "int16": "DataType::INT16",
        "int32": "DataType::INT32",
        "int64": "DataType::INT64",
        "uint64": "DataType::UINT64",
    }
    result = mapping.get(str(dtype))
    if result is None:
        raise RuntimeError(f"Unsupported distributed tensor dtype {dtype}")
    return result


def _generate_local_phase_source(
    *,
    program: _ir_core.Program,
    local_func: _ir_core.Function,
    orch_func: str,
    kernel_name_to_id: dict[str, int],
    kernel_core_type_by_name: dict[str, str],
) -> str:
    ext_refs: dict[str, str] = {}
    value_refs: dict[str, str] = {}
    create_infos: dict[str, str] = {}
    producer_submit_by_var: dict[str, str] = {}
    submit_counter = 0
    lines = [
        '__attribute__((visibility("default")))',
        (
            f"void {orch_func}(PTO2Runtime* rt, const ChipStorageTaskArgs& orch_args, "
            "int orch_thread_num, int orch_thread_index) {"
        ),
        "    (void)orch_thread_num;",
        "    (void)orch_thread_index;",
        "",
        "    // External tensors",
    ]

    tensor_params = [param for param in local_func.params if isinstance(param.type, _ir_core.TensorType)]
    for idx, param in enumerate(tensor_params):
        ext_name = f"ext_{param.name_hint}"
        ext_refs[param.name_hint] = ext_name
        value_refs[param.name_hint] = ext_name
        lines.append(f"    Tensor {ext_name} = from_tensor_arg(orch_args.tensor({idx}));")

    lines.extend(["", "    PTO2_SCOPE(rt) {"])
    stmts = _ir_core.flatten_to_stmts(local_func.body)
    for stmt in stmts:
        if isinstance(stmt, _ir_core.AssignStmt) and isinstance(stmt.value, _ir_core.Call):
            target_name = stmt.var.name_hint
            call = stmt.value
            if isinstance(call.op, _ir_core.Op) and call.op.name == "tensor.create":
                if not isinstance(stmt.var.type, _ir_core.TensorType):
                    raise RuntimeError(f"tensor.create target '{target_name}' is not a TensorType")
                create_name = f"{target_name}_ci"
                shape_values = [_const_int_value(dim) for dim in stmt.var.type.shape]
                shape_list = ", ".join(str(dim) for dim in shape_values)
                lines.append(
                    f"        uint32_t {create_name}_shapes[{len(shape_values)}] = {{{shape_list}}};"
                )
                lines.append(
                    f"        TensorCreateInfo {create_name}("
                    f"{create_name}_shapes, {len(shape_values)}, {_tensor_type_to_runtime_dtype(stmt.var.type.dtype)});"
                )
                create_infos[target_name] = create_name
                lines.append("")
                continue

            if not isinstance(call.op, _ir_core.GlobalVar):
                raise RuntimeError(
                    f"Distributed local phase '{local_func.name}' only supports direct kernel calls, got {call.op}"
                )

            callee_name = call.op.name
            callee_func = _resolve_function(program, callee_name)
            kernel_func_id = kernel_name_to_id.get(callee_name)
            if kernel_func_id is None:
                kernel_func_id = max(kernel_name_to_id.values(), default=-1) + 1
                kernel_name_to_id[callee_name] = kernel_func_id
            core_type = kernel_core_type_by_name.get(callee_name)
            if core_type is None:
                core_type = _core_type_to_config_string(_codegen_core.infer_function_core_type(callee_func))
                kernel_core_type_by_name[callee_name] = core_type
            submit_fn = "pto2_rt_submit_aiv_task" if core_type == "aiv" else "pto2_rt_submit_aic_task"
            submit_name = f"r{submit_counter}"
            submit_counter += 1

            lines.append(f"        // Task {kernel_func_id}: {callee_name}")
            lines.append(f"        Arg params_{submit_name};")

            call_args = list(call.args)
            output_arg_name = call_args[-1].name_hint if call_args else None
            for arg in call_args[:-1]:
                arg_name = arg.name_hint
                ref_name = value_refs.get(arg_name)
                if ref_name is None:
                    raise RuntimeError(f"Unknown input tensor '{arg_name}' in distributed local phase '{local_func.name}'")
                lines.append(f"        params_{submit_name}.add_input({ref_name});")

            if output_arg_name is None:
                raise RuntimeError(f"Kernel call '{callee_name}' in distributed local phase '{local_func.name}' has no args")
            output_ref = value_refs.get(output_arg_name)
            if output_ref is not None and output_arg_name in ext_refs:
                lines.append(f"        params_{submit_name}.add_inout({output_ref});")
            else:
                create_name = create_infos.get(output_arg_name)
                if create_name is None:
                    raise RuntimeError(
                        f"Output tensor '{output_arg_name}' in distributed local phase '{local_func.name}' "
                        "must come from tensor.create or an external parameter"
                    )
                lines.append(f"        params_{submit_name}.add_output({create_name});")

            lines.append(
                f"        SubmitResult {submit_name} = {submit_fn}(rt, {kernel_func_id}, params_{submit_name});"
            )
            deps_added: set[str] = set()
            for arg in call_args[:-1]:
                producer = producer_submit_by_var.get(arg.name_hint)
                if producer is None or producer in deps_added:
                    continue
                lines.append(f"        pto2_rt_add_dependency(rt, {producer}.task_id, {submit_name}.task_id);")
                deps_added.add(producer)

            if output_arg_name in ext_refs:
                value_refs[target_name] = ext_refs[output_arg_name]
                lines.append(f"        const Tensor& {target_name} = {ext_refs[output_arg_name]};")
            else:
                value_refs[target_name] = target_name
                lines.append(f"        const Tensor& {target_name} = {submit_name}.outputs.get_ref(0);")
            producer_submit_by_var[target_name] = submit_name
            lines.append("")
        elif isinstance(stmt, _ir_core.ReturnStmt):
            continue
        else:
            raise RuntimeError(
                f"Distributed local phase '{local_func.name}' only supports straight-line Assign/Return statements"
            )

    lines.extend(["    }", "}"])
    return "\n".join(lines)


def generate(
    transformed_program: _ir_core.Program,
    output_dir: str,
    skip_ptoas: bool = False,
) -> dict[str, str]:
    """Generate all PTO backend output files (kernels + orchestration + config).

    Analogous to the previous codegen pipeline — returns a complete file map for the
    PTO backend. Kernel InCore functions go through the ptoas pipeline by default;
    when ``skip_ptoas=True``, the raw MLIR (.pto) content is returned directly
    without invoking ptoas.

    Args:
        transformed_program: Program after pass pipeline
        output_dir: Base output directory (used for ptoas intermediates when skip_ptoas=False)
        skip_ptoas: When True, skip the ptoas compilation step and return raw MLIR
            content in result_files with .pto extension instead of compiled .cpp wrappers.

    Returns:
        Dict mapping relative file paths to their content.
    """
    result_files: dict[str, str] = {}
    errors: list[tuple[str, Exception]] = []
    orch_funcs = [
        func for func in transformed_program.functions.values() if func.func_type == _ir_core.FunctionType.Orchestration
    ]
    primary_orch = orch_funcs[0] if orch_funcs else None
    distributed_meta = lookup_distributed_program(transformed_program)
    groups, ungrouped = _build_group_mapping(transformed_program)
    kernel_core_type_by_name: dict[str, str] = {}

    def _record_kernel(func_name: str, core_type: _ir_core.CoreType) -> None:
        kernel_core_type_by_name[func_name] = _core_type_to_config_string(core_type)

    def _codegen_single_function(func: _ir_core.Function, pto_code: str) -> None:
        """Compile one InCore function's MLIR through ptoas and emit its wrapper."""
        core_type = _codegen_core.infer_function_core_type(func)
        ct_str = _core_type_to_config_string(core_type)
        _record_kernel(func.name, core_type)

        if skip_ptoas:
            kernel_rel = os.path.join("kernels", ct_str, f"{func.name}.pto")
            result_files[kernel_rel] = pto_code
            return

        ptoas_dir = os.path.join(output_dir, "ptoas")
        os.makedirs(ptoas_dir, exist_ok=True)

        pto_path = os.path.join(ptoas_dir, f"{func.name}.pto")
        with open(pto_path, "w") as f:
            f.write(pto_code)

        cpp_path = os.path.join(ptoas_dir, f"{func.name}.cpp")
        _run_ptoas(
            pto_path,
            cpp_path,
            ptoas_flags=[
                "--enable-insert-sync",
                "--pto-level=level3",
            ]
            + (
                ["--pto-arch", "a5"]
                if _backend_core.get_backend_type() == _backend_core.BackendType.Ascend950
                else []
            ),
        )

        with open(cpp_path) as f:
            ptoas_cpp = f.read()

        wrapper_code = _generate_kernel_wrapper(func, ptoas_cpp)
        kernel_rel = os.path.join("kernels", ct_str, f"{func.name}.cpp")
        result_files[kernel_rel] = wrapper_code

    # Grouped functions: one MLIR module per group
    for group_name, members in groups.items():
        try:
            grouped_program = _ir_core.Program(members, group_name, transformed_program.span)
            codegen_instance = _codegen_core.PTOCodegen()
            pto_code = codegen_instance.generate(grouped_program)

            if skip_ptoas:
                kernel_rel = os.path.join("kernels", f"{group_name}.pto")
                result_files[kernel_rel] = pto_code
                for func in members:
                    _record_kernel(func.name, _codegen_core.infer_function_core_type(func))
            else:
                ptoas_dir = os.path.join(output_dir, "ptoas")
                os.makedirs(ptoas_dir, exist_ok=True)

                pto_path = os.path.join(ptoas_dir, f"{group_name}.pto")
                with open(pto_path, "w") as f:
                    f.write(pto_code)

                cpp_path = os.path.join(ptoas_dir, f"{group_name}.cpp")
                _run_ptoas(
                    pto_path,
                    cpp_path,
                    ptoas_flags=[
                        "--enable-insert-sync",
                        "--pto-level=level3",
                    ]
                    + (
                        ["--pto-arch", "a5"]
                        if _backend_core.get_backend_type() == _backend_core.BackendType.Ascend950
                        else []
                    ),
                )

                with open(cpp_path) as f:
                    ptoas_cpp = f.read()

                for func in members:
                    core_type = _codegen_core.infer_function_core_type(func)
                    _record_kernel(func.name, core_type)
                    wrapper_code = _generate_kernel_wrapper(func, ptoas_cpp)
                    ct_str = _core_type_to_config_string(core_type)
                    kernel_rel = os.path.join("kernels", ct_str, f"{func.name}.cpp")
                    result_files[kernel_rel] = wrapper_code
        except Exception as e:
            func_names = ", ".join(m.name for m in members)
            logger.error("Failed to compile group '%s' [%s]: %s", group_name, func_names, e)
            errors.append((group_name, e))

    # Ungrouped functions: one MLIR module per function (existing behavior)
    for func in ungrouped:
        try:
            single_program = _ir_core.Program([func], func.name, transformed_program.span)
            codegen_instance = _codegen_core.PTOCodegen()
            pto_code = codegen_instance.generate(single_program)
            _codegen_single_function(func, pto_code)
        except Exception as e:
            logger.error("Failed to compile function '%s': %s", func.name, e)
            errors.append((func.name, e))

    if distributed_meta is not None:
        legacy_source_mode = (
            distributed_meta.orchestration_source_fn is not None or bool(distributed_meta.external_kernels)
        )
        phase_specs_for_runtime: list[dict[str, object]] | None = None

        if legacy_source_mode:
            kernel_name_to_id = {name: idx for idx, name in enumerate(sorted(kernel_core_type_by_name))}
            next_func_id = len(kernel_name_to_id)
            for external_kernel in distributed_meta.external_kernels:
                ct_str = str(external_kernel.core_type)
                if ct_str not in {"aic", "aiv"}:
                    errors.append(
                        (external_kernel.name, ValueError(f"Unsupported external kernel core_type '{ct_str}'"))
                    )
                    continue
                kernel_name_to_id[external_kernel.name] = next_func_id
                kernel_core_type_by_name[external_kernel.name] = ct_str
                next_func_id += 1
            compile_ctx = DistributedCompileContext(
                program_name=transformed_program.name,
                func_name_to_id=dict(kernel_name_to_id),
                func_name_to_core_type=dict(kernel_core_type_by_name),
            )

            try:
                assert distributed_meta.orchestration_source_fn is not None
                orch_source = call_source_fn(distributed_meta.orchestration_source_fn, compile_ctx)
                result_files[f"orchestration/{distributed_meta.orchestration_source_name}.cpp"] = orch_source
            except Exception as e:
                logger.error("Failed to generate distributed orchestration for '%s': %s", transformed_program.name, e)
                errors.append((f"{transformed_program.name}:distributed_orchestration", e))

            for external_kernel in distributed_meta.external_kernels:
                try:
                    kernel_source = call_source_fn(external_kernel.source_fn, compile_ctx)
                    ct_str = str(external_kernel.core_type)
                    result_files[f"kernels/{ct_str}/{external_kernel.name}.cpp"] = kernel_source
                except Exception as e:
                    logger.error("Failed to generate external kernel '%s': %s", external_kernel.name, e)
                    errors.append((external_kernel.name, e))
        else:
            kernel_name_to_id: dict[str, int] = {}
            phase_functions: list[str] = []
            phase_specs_for_runtime = []

            local_phases = [phase for phase in distributed_meta.phases if isinstance(phase, DistributedLocalPhase)]
            if len(local_phases) > 1:
                errors.append(
                    (
                        f"{transformed_program.name}:distributed_local_phases",
                        RuntimeError("Multiple DistributedLocalPhase entries are not supported yet"),
                    )
                )
            else:
                for phase in distributed_meta.phases:
                    if isinstance(phase, DistributedLocalPhase):
                        try:
                            local_func = _resolve_function(transformed_program, phase.local_func)
                            if local_func.func_type != _ir_core.FunctionType.Orchestration:
                                raise RuntimeError(
                                    f"DistributedLocalPhase '{phase.local_func}' must target an orchestration function"
                                )
                            phase_functions.append(
                                _generate_local_phase_source(
                                    program=transformed_program,
                                    local_func=local_func,
                                    orch_func=phase.orch_func,
                                    kernel_name_to_id=kernel_name_to_id,
                                    kernel_core_type_by_name=kernel_core_type_by_name,
                                )
                            )
                            phase_args = phase.args or _default_phase_args_from_function(local_func)
                            phase_specs_for_runtime.append(
                                {
                                    "orch_func": phase.orch_func,
                                    "barrier_before": bool(phase.barrier_before),
                                    "args": _serialize_phase_args(phase_args),
                                }
                            )
                        except Exception as e:
                            logger.error(
                                "Failed to generate distributed local phase '%s': %s", phase.local_func, e
                            )
                            errors.append((f"{transformed_program.name}:{phase.local_func}", e))
                    elif isinstance(phase, DistributedAllReducePhase):
                        try:
                            kernel_name = phase.resolved_kernel_name()
                            if kernel_name in kernel_name_to_id:
                                raise RuntimeError(f"Duplicate distributed kernel name '{kernel_name}'")
                            next_func_id = max(kernel_name_to_id.values(), default=-1) + 1
                            kernel_name_to_id[kernel_name] = next_func_id
                            ct_str = str(phase.core_type)
                            if ct_str not in {"aic", "aiv"}:
                                raise RuntimeError(f"Unsupported allreduce core type '{ct_str}'")
                            kernel_core_type_by_name[kernel_name] = ct_str
                            input_type = _resolve_tensor_type_by_name(transformed_program, phase.input_name)
                            shape = [_const_int_value(dim) for dim in input_type.shape]
                            runtime_dtype = _tensor_type_to_runtime_dtype(input_type.dtype)
                            result_files[f"kernels/{ct_str}/{kernel_name}.cpp"] = _generate_allreduce_kernel_source(
                                int(phase.chunk_elems)
                            )
                            phase_functions.append(
                                _generate_allreduce_phase_source(
                                    orch_func=phase.orch_func,
                                    kernel_func_id=next_func_id,
                                    shape=shape,
                                    dtype=runtime_dtype,
                                )
                            )
                            phase_specs_for_runtime.append(
                                {
                                    "orch_func": phase.orch_func,
                                    "barrier_before": bool(phase.barrier_before),
                                    "args": _serialize_phase_args(
                                        [
                                            PhaseArg(phase.input_name, "scalar"),
                                            PhaseArg(phase.output_name, "scalar"),
                                            PhaseArg(phase.nranks_arg, "scalar"),
                                            PhaseArg(phase.root_arg, "scalar"),
                                            PhaseArg(phase.device_ctx_arg, "scalar"),
                                        ]
                                    ),
                                }
                            )
                        except Exception as e:
                            logger.error("Failed to generate distributed allreduce phase '%s': %s", phase.orch_func, e)
                            errors.append((f"{transformed_program.name}:{phase.orch_func}", e))
                    elif isinstance(phase, DistributedPhase):
                        phase_entry = {
                            "orch_func": phase.orch_func,
                            "barrier_before": bool(phase.barrier_before),
                        }
                        args = _serialize_phase_args(phase.args)
                        if args is not None:
                            phase_entry["args"] = args
                        phase_specs_for_runtime.append(phase_entry)
                    else:
                        errors.append(
                            (
                                f"{transformed_program.name}:distributed_phase",
                                RuntimeError(f"Unsupported distributed phase type {type(phase).__name__}"),
                            )
                        )

            if phase_functions:
                result_files[f"orchestration/{distributed_meta.orchestration_source_name}.cpp"] = (
                    _generate_distributed_phase_file(phase_functions)
                )

        if not skip_ptoas and distributed_meta.phases:
            runtime_cfg = None
            runtime_env = None
            if distributed_meta.runtime is not None:
                runtime_cfg = {
                    "runtime": distributed_meta.runtime.runtime,
                    "aicpu_thread_num": distributed_meta.runtime.aicpu_thread_num,
                    "orch_thread_num": distributed_meta.runtime.orch_thread_num,
                    "block_dim": distributed_meta.runtime.block_dim,
                }
                runtime_env = dict(distributed_meta.runtime.env)

            kernel_entries = [
                {
                    "name": name,
                    "func_id": func_id,
                    "core_type": kernel_core_type_by_name[name],
                }
                for name, func_id in sorted(kernel_name_to_id.items(), key=lambda item: int(item[1]))
            ]
            result_files["kernel_config.py"] = _generate_config_file(
                orch_source_name=distributed_meta.orchestration_source_name,
                orch_function_name=distributed_meta.phases[0].orch_func,
                kernel_entries=kernel_entries,
                runtime_config=runtime_cfg,
                runtime_env=runtime_env,
                distributed_spec=_build_distributed_spec(distributed_meta, phase_specs_for_runtime),
            )
    elif primary_orch is not None:
        try:
            orch_result = _codegen_core.generate_orchestration(transformed_program, primary_orch)
            result_files[f"orchestration/{primary_orch.name}.cpp"] = (
                f"// Orchestration Function: {primary_orch.name}\n"
                f"// Generated by PyPTO IR Compiler\n\n"
                f"{orch_result.code}"
            )
            if not skip_ptoas:
                kernel_entries = [
                    {
                        "name": name,
                        "func_id": func_id,
                        "core_type": _core_type_to_config_string(orch_result.func_name_to_core_type[name]),
                    }
                    for name, func_id in orch_result.func_name_to_id.items()
                ]
                result_files["kernel_config.py"] = _generate_config_file(
                    orch_source_name=primary_orch.name,
                    orch_function_name="aicpu_orchestration_entry",
                    kernel_entries=kernel_entries,
                )
        except Exception as e:
            logger.error("Failed to generate orchestration '%s': %s", primary_orch.name, e)
            errors.append((primary_orch.name, e))

    if errors:
        report = _format_error_report(errors, output_dir)
        if result_files:
            raise PartialCodegenError(report, result_files)
        raise RuntimeError(report)

    return result_files
