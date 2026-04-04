# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for distributed C++ code generation."""

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir, passes
from pypto.backend import BackendType


class TestDistributedCodegen:
    """Test distributed C++ codegen on outlined hierarchy programs."""

    def test_worker_and_orchestrator_codegen(self):
        """Outlined hierarchy program produces correct C++ structure.

        Input: a program with one HOST WORKER scope called from an
        orchestrator-level function. After outlining, the codegen should
        emit:
        - A static void worker function
        - A submit_worker call at the call site
        - Correct includes and namespace structure
        """

        @pl.program
        class Input:
            @pl.function(level=pl.Level.POD, role=pl.Role.Orchestrator)
            def pod_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        # Run prerequisite passes
        program = passes.convert_to_ssa()(Input)
        program = passes.outline_hierarchy_scopes()(program)

        # Run distributed codegen
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # --- Verify includes ---
        assert '#include "core/tensor.h"' in code
        assert '#include "runtime/level_runtime.h"' in code
        assert '#include "runtime/tree_reduce.h"' in code

        # --- Verify worker function ---
        assert "static void pod_orch_host_worker_0" in code

        # --- Verify orchestrator function ---
        assert "static LinquTensor pod_orch" in code
        assert "LevelRuntime&" in code

        # --- Verify call-site lowering ---
        assert "submit_worker" in code
        assert '"pod_orch_host_worker_0"' in code

        # --- Verify ordering ---
        # Worker must appear before orchestrator (dependency order)
        worker_pos = code.index("static void pod_orch_host_worker_0")
        orch_pos = code.index("static LinquTensor pod_orch")
        assert worker_pos < orch_pos

    def test_multi_level_orchestrators(self):
        """Multiple orchestrator levels produce correct runtime variables.

        Tests that L4 (POD) and L5 (CLOS1) orchestrators use rt_l4 and
        rt_l5 respectively in their submit calls.
        """

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def leaf_worker(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(level=pl.Level.POD, role=pl.Role.Orchestrator)
            def pod_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.leaf_worker(x)
                return y

            @pl.function(level=pl.Level.CLOS1, role=pl.Role.Orchestrator)
            def clos1_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.pod_orch(x)
                return y

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # Worker is a static void
        assert "static void leaf_worker" in code

        # Pod orchestrator calls submit_worker on rt_l3
        # (because leaf_worker is at HOST = level 3)
        assert "rt_l3" in code
        assert "submit_worker" in code

        # Clos1 orchestrator calls submit_orchestrator on rt_l4
        # (because pod_orch is at POD = level 4)
        assert "rt_l4" in code
        assert "submit_orchestrator" in code

    def test_for_loop_codegen(self):
        """ForStmt in function body produces C++ for loop."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.POD, role=pl.Role.Orchestrator)
            def orch_with_loop(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(0, 4):
                    y = pl.add(y, x)
                return y

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "for (int" in code
        assert "< 4" in code

    def test_using_declarations(self):
        """Generated code contains required using declarations."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def simple_worker(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "using linqu::LinquTensor;" in code
        assert "using linqu::LevelRuntime;" in code

    def test_anonymous_namespace(self):
        """Internal functions are wrapped in anonymous namespace."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def simple_worker(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "namespace {" in code
        assert "}  // namespace" in code

    def test_topology_constants(self):
        """Topology constants emitted for used levels."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def worker(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # HOST = level 3, so we expect kNumL3
        assert "kNumL3" in code
        assert "env_int" in code

    def test_distributed_phase_codegen_without_embedded_cpp(self, tmp_path):
        """Declarative distributed phases generate orchestration and collective code in backend."""

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class DistributedProgram:
            DISTRIBUTED = pl.DistributedProgram(
                phases=[
                    pl.DistributedLocalPhase(local_func="local_phase", orch_func="dist_phase1"),
                    pl.DistributedAllReducePhase(
                        input_name="partial_out",
                        output_name="output",
                        orch_func="dist_phase2",
                        barrier_before=True,
                    ),
                ],
                orchestration_source_name="dist_orch",
                buffer_attrs=[pl.BufferAttr("partial_out", placement="window", data_prefix_elems=4)],
                inputs=["x", "w"],
                outputs=["output"],
            )

            @pl.function(type=pl.FunctionType.InCore)
            def vec_add(
                self,
                x: pl.Tensor[[4, 16], pl.FP32],
                w: pl.Tensor[[4, 16], pl.FP32],
                partial_out: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                x_tile = pl.load(x, [0, 0], [4, 16])
                w_tile = pl.load(w, [0, 0], [4, 16])
                out_tile = pl.add(x_tile, w_tile)
                return pl.store(out_tile, [0, 0], partial_out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def local_phase(
                self,
                x: pl.Tensor[[4, 16], pl.FP32],
                w: pl.Tensor[[4, 16], pl.FP32],
                partial_out: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                partial_out = self.vec_add(x, w, partial_out)
                return partial_out

        out_dir = ir.compile(
            DistributedProgram,
            output_dir=str(tmp_path),
            dump_passes=False,
            skip_ptoas=True,
            backend_type=BackendType.Ascend910B,
        )

        dist_orch = (tmp_path / "orchestration" / "dist_orch.cpp").read_text(encoding="utf-8")
        allreduce_kernel = (tmp_path / "kernels" / "aiv" / "dist_phase2_kernel.cpp").read_text(encoding="utf-8")

        assert out_dir == str(tmp_path)
        assert "void dist_phase1(PTO2Runtime* rt" in dist_orch
        assert "void dist_phase2(" in dist_orch
        assert "PTO2_SCOPE(rt)" in dist_orch
        assert "pto2_rt_submit_aiv_task(rt, " in dist_orch
        assert "from_tensor_arg(orch_args.tensor(0));" in dist_orch
        assert "reinterpret_cast<void*>(orch_args.scalar(0))" in dist_orch
        assert "reinterpret_cast<void*>(orch_args.scalar(1))" in dist_orch
        assert "uint64_t nranks = orch_args.scalar(2);" in dist_orch
        assert "uint64_t root = orch_args.scalar(3);" in dist_orch
        assert "uint64_t device_ctx = orch_args.scalar(4);" in dist_orch
        assert "input_tensor->ndims == 0 || output_tensor->ndims != input_tensor->ndims" in allreduce_kernel
        assert "output_tensor->shapes[i] != input_tensor->shapes[i]" in allreduce_kernel
        assert "total_elems *= static_cast<uint64_t>(input_tensor->shapes[i]);" in allreduce_kernel
        assert "ALLREDUCE_CHUNK = 256" in allreduce_kernel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
