"""Program-level distributed metadata helpers for PyPTO examples and runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from pypto.pypto_core.ir import Program


@dataclass(frozen=True)
class BufferAttr:
    """Per-buffer distributed placement and backing-storage metadata."""

    name: str
    placement: str | None = None
    data_prefix_elems: int = 0


@dataclass(frozen=True)
class PhaseArg:
    """One distributed orchestration argument token."""

    token: str
    kind: str = "scalar"

    def __post_init__(self) -> None:
        if self.kind not in {"scalar", "tensor"}:
            raise ValueError(f"Unsupported phase arg kind {self.kind!r}, expected 'scalar' or 'tensor'")


def tensor_arg(token: str) -> PhaseArg:
    """Return a tensor-valued orchestration argument token."""

    return PhaseArg(token=token, kind="tensor")


def scalar_arg(token: str) -> PhaseArg:
    """Return a scalar-valued orchestration argument token."""

    return PhaseArg(token=token, kind="scalar")


@dataclass(frozen=True)
class DistributedPhase:
    """Legacy phase entry used by raw-source distributed orchestration."""

    orch_func: str
    barrier_before: bool = False
    args: list[PhaseArg] | None = None


@dataclass(frozen=True)
class DistributedLocalPhase:
    """Compile a PyPTO orchestration function as one distributed phase."""

    local_func: str
    orch_func: str
    barrier_before: bool = False
    args: list[PhaseArg] | None = None


@dataclass(frozen=True)
class DistributedAllReducePhase:
    """Built-in window-backed float32 allreduce(sum) phase."""

    input_name: str
    output_name: str
    orch_func: str
    barrier_before: bool = False
    nranks_arg: str = "nranks"
    root_arg: str = "root"
    device_ctx_arg: str = "deviceCtx"
    kernel_name: str | None = None
    core_type: str = "aiv"
    chunk_elems: int = 256

    def resolved_kernel_name(self) -> str:
        return self.kernel_name or f"{self.orch_func}_kernel"


@dataclass(frozen=True)
class DistributedRuntime:
    """Runtime selection and environment for distributed execution."""

    runtime: str = "aicpu_build_graph"
    aicpu_thread_num: int = 1
    orch_thread_num: int = 0
    block_dim: int = 1
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DistributedCompileContext:
    """Compilation context exposed to external source generators."""

    program_name: str
    func_name_to_id: dict[str, int]
    func_name_to_core_type: dict[str, str]


SourceFn = Callable[[DistributedCompileContext], str] | Callable[[], str]


@dataclass(frozen=True)
class ExternalKernel:
    """Externally-authored kernel source compiled as part of a distributed program."""

    name: str
    core_type: str
    source_fn: SourceFn


DistributedPhaseSpec = DistributedPhase | DistributedLocalPhase | DistributedAllReducePhase


@dataclass(frozen=True)
class DistributedProgram:
    """Program-level distributed execution metadata."""

    phases: list[DistributedPhaseSpec]
    orchestration_source_fn: SourceFn | None = None
    orchestration_source_name: str = "distributed_orchestration"
    buffer_attrs: list[BufferAttr] = field(default_factory=list)
    external_kernels: list[ExternalKernel] = field(default_factory=list)
    args: list[str] | None = None
    inputs: list[str] | None = None
    outputs: list[str] | None = None
    win_sync_prefix: int = 256
    runtime: DistributedRuntime | None = None
    comm_include_dirs: list[str] = field(default_factory=list)


_PROGRAM_DISTRIBUTED_METADATA: dict[tuple[str, str, int, int], DistributedProgram] = {}


def _program_key(program: Program | Any) -> tuple[str, str, int, int]:
    span = getattr(program, "span", None)
    filename = getattr(span, "filename", "") if span is not None else ""
    begin_line = int(getattr(span, "begin_line", 0) or 0) if span is not None else 0
    begin_column = int(getattr(span, "begin_column", 0) or 0) if span is not None else 0
    name = str(getattr(program, "name", ""))
    return (name, filename, begin_line, begin_column)


def register_distributed_program(program: Program | Any, metadata: DistributedProgram) -> None:
    """Register distributed metadata for a program produced by ``@pl.program``."""

    _PROGRAM_DISTRIBUTED_METADATA[_program_key(program)] = metadata


def lookup_distributed_program(program: Program | Any) -> DistributedProgram | None:
    """Return distributed metadata for *program* when present."""

    return _PROGRAM_DISTRIBUTED_METADATA.get(_program_key(program))


def call_source_fn(source_fn: SourceFn, ctx: DistributedCompileContext) -> str:
    """Invoke a source generator that may or may not accept a compile context."""

    try:
        return source_fn(ctx)  # type: ignore[misc]
    except TypeError:
        return source_fn()  # type: ignore[misc]
