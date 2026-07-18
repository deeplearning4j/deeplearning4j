# ******************************************************************************
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************
"""Tests for the high-level Python-idiomatic API added in sdx_runtime:
InputMetadata, ExecutionSummary, SdxContext.get_inputs(), SdxContext.run_named().

These tests exercise pure Python logic only — no native library required.
"""

import ctypes
import pathlib
import sys
import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_PARENT = _THIS_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from sdx_runtime import (
    DynamicOutput,
    ExecutionReport,
    ExecutionSummary,
    InputMetadata,
    RunOptions,
    SDX_GPU_TARGET_METAL,
    SDX_GPU_TARGET_VULKAN,
    SdxContext,
    SdxModel,
    SdxRuntime,
    TensorView,
)


class GpuTargetConstantsTests(unittest.TestCase):
    def test_mobile_gpu_targets_are_public(self):
        self.assertEqual(SDX_GPU_TARGET_VULKAN, 3)
        self.assertEqual(SDX_GPU_TARGET_METAL, 4)


# ---------------------------------------------------------------------------
# Helpers: build a minimal SdxContext wired to a mock C library, no dlopen.
# ---------------------------------------------------------------------------

def _make_mock_context(input_names_list, plan_phase=2, exec_count=3):
    """Return (runtime, context) pair backed by a MagicMock C library."""
    mock_lib = MagicMock()

    # sdxGetNumInputs
    mock_lib.sdxGetNumInputs.return_value = len(input_names_list)

    # sdxGetInputName: positional dispatch
    def _get_input_name(_ctx_handle, idx):
        if 0 <= idx < len(input_names_list):
            return input_names_list[idx].encode("utf-8")
        return None

    mock_lib.sdxGetInputName.side_effect = _get_input_name

    # sdxGetNumOutputs
    mock_lib.sdxGetNumOutputs.return_value = 1

    # sdxGetPlanPhase
    mock_lib.sdxGetPlanPhase.return_value = plan_phase

    # sdxGetExecutionCount
    mock_lib.sdxGetExecutionCount.return_value = exec_count

    # Caller-allocated and runtime-allocated execution — success
    mock_lib.sdxRun.return_value = 0
    mock_lib.sdxRunAllocating.return_value = 0

    # sdxGetExecutionReport — fill in a synthetic report
    def _fill_report(_ctx_handle, report_ptr):
        r = report_ptr._obj
        r.status_code = 0
        r.requested_backend = 0
        r.applied_backend = 1
        r.used_fallback = 0
        r.plan_phase = plan_phase
        r.execution_count = exec_count
        r.execution_time_ns = 500_000  # 0.5 ms
        r.requested_gpu_target = 0
        r.applied_gpu_target = 0
        return 0

    mock_lib.sdxGetExecutionReport.side_effect = _fill_report
    mock_lib.sdxGetLastError.return_value = None

    # Build the minimal object graph without touching dlopen.
    runtime = object.__new__(SdxRuntime)
    runtime._lib = mock_lib

    from sdx_runtime import _Handles
    import ctypes
    runtime._handles = _Handles(runtime=ctypes.c_void_p(1))

    ctx = object.__new__(SdxContext)
    ctx._runtime = runtime
    import ctypes as _ct
    ctx._context_handle = _ct.c_void_p(2)

    return runtime, ctx


class ParameterBoundContextTests(unittest.TestCase):
    def test_inference_context_uses_options_and_retains_model(self):
        mock_lib = MagicMock()

        def _create(_model, _outputs, _count, options_ptr, out_context):
            self.assertEqual(options_ptr._obj.bind_model_parameters, 1)
            out_context._obj.value = 4
            return 0

        mock_lib.sdxCreateContextWithOptions.side_effect = _create
        mock_lib.sdxGetLastError.return_value = None

        runtime = object.__new__(SdxRuntime)
        runtime._lib = mock_lib
        from sdx_runtime import _Handles
        runtime._handles = _Handles(runtime=ctypes.c_void_p(1))

        model = object.__new__(SdxModel)
        model._runtime = runtime
        model._model_handle = ctypes.c_void_p(3)

        context = model.create_inference_context(["output"])
        self.assertIs(context._model, model)
        self.assertEqual(context._context_handle.value, 4)
        mock_lib.sdxCreateContext.assert_not_called()


class BundleAssetMetadataTests(unittest.TestCase):
    def test_model_exposes_resolved_offline_assets(self):
        mock_lib = MagicMock()
        mock_lib.sdxGetTokenizerPath.return_value = b"/bundle/tokenizer.json"
        mock_lib.sdxGetTextGenerationConfigPath.return_value = (
            b"/bundle/text-generation.json"
        )

        runtime = object.__new__(SdxRuntime)
        runtime._lib = mock_lib
        from sdx_runtime import _Handles
        runtime._handles = _Handles(runtime=ctypes.c_void_p(1))

        model = object.__new__(SdxModel)
        model._runtime = runtime
        model._model_handle = ctypes.c_void_p(3)

        self.assertEqual(model.tokenizer_path, "/bundle/tokenizer.json")
        self.assertEqual(
            model.text_generation_config_path, "/bundle/text-generation.json"
        )


class DynamicOutputTests(unittest.TestCase):
    def test_numpy_converts_owned_bytes(self):
        expected = np.arange(4, dtype=np.float32).reshape(2, 2)
        output = DynamicOutput(
            data=expected.tobytes(), shape=(2, 2), dtype=5, name="logits"
        )
        np.testing.assert_array_equal(output.numpy(), expected)

    def test_run_allocating_copies_runtime_output(self):
        runtime, ctx = _make_mock_context(["input_ids"])
        runtime._lib.sdxGetOutputName.return_value = b"logits"
        values = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
        shape = (ctypes.c_int64 * 2)(2, 2)

        def _fill_output(_context, index, output_ptr):
            self.assertEqual(index, 0)
            view = output_ptr._obj
            view.data = ctypes.cast(values, ctypes.c_void_p)
            view.shape = ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64))
            view.rank = 2
            view.dtype = 5
            view.bytes = ctypes.sizeof(values)
            view.device_type = 0
            view.device_id = -1
            return 0

        runtime._lib.sdxGetOutputTensor.side_effect = _fill_output
        outputs = ctx.run_allocating(
            [np.asarray([[7]], dtype=np.int32)],
            options=RunOptions(gpu_target=SDX_GPU_TARGET_VULKAN),
        )

        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(outputs[0], DynamicOutput)
        self.assertEqual(outputs[0].name, "logits")
        self.assertEqual(outputs[0].shape, (2, 2))
        np.testing.assert_array_equal(
            outputs[0].numpy(), np.asarray([[1.0, 2.0], [3.0, 4.0]])
        )
        runtime._lib.sdxRunAllocating.assert_called_once()


class InputMetadataTests(unittest.TestCase):
    def test_fields_frozen(self):
        meta = InputMetadata(name="x", index=3)
        self.assertEqual(meta.name, "x")
        self.assertEqual(meta.index, 3)
        with self.assertRaises(Exception):  # frozen dataclass
            meta.name = "y"  # type: ignore[misc]

    def test_equality(self):
        a = InputMetadata(name="w1", index=0)
        b = InputMetadata(name="w1", index=0)
        self.assertEqual(a, b)

    def test_inequality_index(self):
        a = InputMetadata(name="w1", index=0)
        b = InputMetadata(name="w1", index=1)
        self.assertNotEqual(a, b)


class ExecutionSummaryTests(unittest.TestCase):
    def _make_summary(self, **overrides):
        defaults = dict(
            status_code=0,
            requested_backend=0,
            applied_backend=1,
            used_fallback=False,
            plan_phase=2,
            execution_count=5,
            execution_time_ns=1_234_567,
            requested_gpu_target=0,
            applied_gpu_target=0,
        )
        defaults.update(overrides)
        return ExecutionSummary(**defaults)

    def test_execution_time_ms(self):
        s = self._make_summary(execution_time_ns=1_000_000)
        self.assertAlmostEqual(s.execution_time_ms, 1.0, places=6)

    def test_plan_phase_name_known(self):
        self.assertEqual(self._make_summary(plan_phase=0).plan_phase_name, "SLOT_BY_SLOT")
        self.assertEqual(self._make_summary(plan_phase=1).plan_phase_name, "SHAPES_FROZEN")
        self.assertEqual(self._make_summary(plan_phase=2).plan_phase_name, "REPLAYING")
        self.assertEqual(self._make_summary(plan_phase=3).plan_phase_name, "REPLAY_BLOCKED")

    def test_plan_phase_name_unknown(self):
        self.assertIn("99", self._make_summary(plan_phase=99).plan_phase_name)

    def test_applied_backend_name_known(self):
        self.assertEqual(self._make_summary(applied_backend=0).applied_backend_name, "AUTO")
        self.assertEqual(self._make_summary(applied_backend=5).applied_backend_name, "TRITON")
        self.assertEqual(self._make_summary(applied_backend=9).applied_backend_name, "HIP_GRAPHS")
        self.assertEqual(self._make_summary(applied_backend=11).applied_backend_name, "VULKAN")
        self.assertEqual(self._make_summary(applied_backend=14).applied_backend_name, "HEXAGON")

    def test_applied_backend_name_unknown(self):
        self.assertIn("99", self._make_summary(applied_backend=99).applied_backend_name)

    def test_frozen(self):
        s = self._make_summary()
        with self.assertRaises(Exception):
            s.status_code = 1  # type: ignore[misc]

    def test_from_ctypes_round_trip(self):
        """ExecutionSummary._from_ctypes maps every field correctly."""
        import ctypes

        raw = ExecutionReport()
        raw.status_code = 0
        raw.requested_backend = 0
        raw.applied_backend = 2
        raw.used_fallback = 1
        raw.plan_phase = 2
        raw.execution_count = 7
        raw.execution_time_ns = 9_999_000
        raw.requested_gpu_target = 1
        raw.applied_gpu_target = 1

        s = ExecutionSummary._from_ctypes(raw)
        self.assertEqual(s.status_code, 0)
        self.assertEqual(s.applied_backend, 2)
        self.assertEqual(s.applied_backend_name, "CUDA_GRAPHS")
        self.assertTrue(s.used_fallback)
        self.assertEqual(s.plan_phase, 2)
        self.assertEqual(s.plan_phase_name, "REPLAYING")
        self.assertEqual(s.execution_count, 7)
        self.assertAlmostEqual(s.execution_time_ms, 9.999, places=3)


class SdxContextGetInputsTests(unittest.TestCase):
    def test_get_inputs_returns_metadata_in_order(self):
        _, ctx = _make_mock_context(["w1", "b1", "w2", "b2", "x"])
        inputs = ctx.get_inputs()
        self.assertEqual(len(inputs), 5)
        self.assertIsInstance(inputs[0], InputMetadata)
        self.assertEqual(inputs[0].name, "w1")
        self.assertEqual(inputs[0].index, 0)
        self.assertEqual(inputs[4].name, "x")
        self.assertEqual(inputs[4].index, 4)

    def test_get_inputs_empty(self):
        _, ctx = _make_mock_context([])
        inputs = ctx.get_inputs()
        self.assertEqual(inputs, [])


class SdxContextRunNamedTests(unittest.TestCase):
    def _make_arrays(self):
        return {
            "w1": np.linspace(-1, 1, 32, dtype=np.float32).reshape(4, 8),
            "b1": np.linspace(0, 0.7, 8, dtype=np.float32),
            "w2": np.linspace(1, -1, 24, dtype=np.float32).reshape(8, 3),
            "b2": np.linspace(-0.1, 0.1, 3, dtype=np.float32),
            "x": np.linspace(0.1, 0.8, 8, dtype=np.float32).reshape(2, 4),
        }

    def test_run_named_reorders_and_returns_summary(self):
        # Plan binding order: w1, b1, w2, b2, x
        _, ctx = _make_mock_context(["w1", "b1", "w2", "b2", "x"])
        probs = np.zeros((2, 3), dtype=np.float32)
        feed = self._make_arrays()

        report = ctx.run_named(feed, [probs])

        self.assertIsInstance(report, ExecutionSummary)
        self.assertEqual(report.plan_phase_name, "REPLAYING")
        self.assertEqual(report.applied_backend_name, "SLOT_BY_SLOT")
        self.assertAlmostEqual(report.execution_time_ms, 0.5, places=3)

    def test_run_named_missing_input_raises_value_error(self):
        _, ctx = _make_mock_context(["w1", "x"])
        probs = np.zeros((2, 3), dtype=np.float32)
        feed = {"w1": np.zeros((4, 8), dtype=np.float32)}  # missing "x"

        with self.assertRaises(ValueError) as cm:
            ctx.run_named(feed, [probs])
        self.assertIn("x", str(cm.exception))
        self.assertIn("missing", str(cm.exception))

    def test_run_named_extra_input_raises_value_error(self):
        _, ctx = _make_mock_context(["w1", "x"])
        probs = np.zeros((2, 3), dtype=np.float32)
        feed = {
            "w1": np.zeros((4, 8), dtype=np.float32),
            "x": np.zeros((2, 4), dtype=np.float32),
            "surprise": np.zeros((1,), dtype=np.float32),  # not in plan
        }

        with self.assertRaises(ValueError) as cm:
            ctx.run_named(feed, [probs])
        self.assertIn("surprise", str(cm.exception))
        self.assertIn("unexpected", str(cm.exception))

    def test_run_named_does_not_require_insertion_order(self):
        """Insertion order of feed dict must NOT matter — plan order wins."""
        _, ctx = _make_mock_context(["w1", "b1", "w2", "b2", "x"])
        probs = np.zeros((2, 3), dtype=np.float32)
        # Deliberately reversed insertion order
        feed_reversed = dict(reversed(list(self._make_arrays().items())))

        report = ctx.run_named(feed_reversed, [probs])
        self.assertIsInstance(report, ExecutionSummary)

    def test_run_named_with_run_options(self):
        _, ctx = _make_mock_context(["x"])
        out = np.zeros((2, 3), dtype=np.float32)
        opts = RunOptions(backend=1)
        feed = {"x": np.zeros((2, 4), dtype=np.float32)}

        report = ctx.run_named(feed, [out], options=opts)
        self.assertIsInstance(report, ExecutionSummary)


if __name__ == "__main__":
    unittest.main()
