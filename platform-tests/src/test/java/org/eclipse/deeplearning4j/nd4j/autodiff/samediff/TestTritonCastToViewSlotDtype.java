package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for Triton cast-to-view-slot dtype bug.
 *
 * When a Triton-compiled cast op writes to an output slot that belongs to a
 * view-capable op (e.g. strided_slice), the output slot may not have a
 * pre-allocated array at compile time. Without the fix, the Triton IR builder
 * defaults the output dtype to FLOAT32 instead of using the cast's target
 * dtype from iArgs[0]. This causes the kernel to write FLOAT32 data into an
 * INT64 buffer, and the downstream strided_slice reads garbage values
 * (e.g. end_index = 1593835520).
 *
 * These tests verify that:
 * 1. Cast ops with INT64/INT32 targets produce correct output through DSP
 * 2. StridedSlice receives correct begin/end/strides values from upstream casts
 * 3. The full pipeline (cast → strided_slice → downstream ops) produces correct results
 */
@Slf4j
@DisplayName("Triton Cast-to-ViewSlot Dtype Regression Tests")
public class TestTritonCastToViewSlotDtype {

    private static final double TOL = 1e-5;

    /**
     * Test that a cast to INT64 followed by strided_slice produces correct results.
     * This is the exact pattern from the SmolDocling VLM model that was failing:
     * reshape(shape) → cast(INT64) → strided_slice(data, begin, END, strides)
     *
     * The cast output feeds into strided_slice as the "end" index parameter.
     * Without the dtype fix, the cast kernel writes FLOAT32 to an INT64 buffer,
     * and strided_slice reads garbage end indices like 1593835520.
     */
    @Test
    @DisplayName("Cast INT64 → strided_slice end index: correct value propagation")
    void testCastInt64ToStridedSliceEnd() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Input data: [1, 10] float tensor
        SDVariable data = sd.placeHolder("data", DataType.FLOAT, 1, 10);

        // End index: scalar INT32 that gets cast to INT64
        // In the VLM model, this comes from a reshape of a shape_of output
        SDVariable endIdx = sd.placeHolder("endIdx", DataType.INT32, 1);

        // Cast endIdx from INT32 to INT64 (this is the op that was broken)
        SDVariable endInt64 = sd.castTo("endInt64", endIdx, DataType.INT64);

        // Build strided_slice: data[0:endIdx] along axis 1
        SDVariable begin = sd.constant("begin", Nd4j.createFromArray(new long[]{0, 0}));
        SDVariable strides = sd.constant("strides", Nd4j.createFromArray(new long[]{1, 1}));

        // Create the end array: [1, endInt64]
        SDVariable one = sd.constant("one", Nd4j.createFromArray(new long[]{1}));
        SDVariable endArr = sd.concat("endArr", 0, one, endInt64);

        SDVariable sliced = sd.stridedSlice("sliced", data, begin, endArr, strides);

        // Sum for scalar output
        SDVariable result = sd.sum("result", sliced);

        // Execute with endIdx = 5 → data[0:1, 0:5] → sum of first 5 elements
        INDArray dataArr = Nd4j.linspace(1, 10, 10, DataType.FLOAT).reshape(1, 10);
        INDArray endIdxArr = Nd4j.createFromArray(new int[]{5});

        // Run via standard output first to get expected result
        Map<String, INDArray> expected = sd.output(
                Map.of("data", dataArr, "endIdx", endIdxArr), "result");
        double expectedVal = expected.get("result").getDouble(0);

        // Run via outputDirect (DSP path)
        Map<String, INDArray> dspResult = sd.outputDirect(
                Map.of("data", dataArr, "endIdx", endIdxArr), "result");
        double dspVal = dspResult.get("result").getDouble(0);

        log.info("Expected: {}, DSP: {}", expectedVal, dspVal);
        assertEquals(expectedVal, dspVal, TOL,
                "DSP result should match standard execution for cast INT64 → strided_slice");

        sd.close();
    }

    /**
     * Test cast to INT64 with simple downstream consumption via DSP.
     * Verifies the cast itself produces correct output dtype and value,
     * independent of strided_slice.
     */
    @Test
    @DisplayName("Cast INT32→INT64 produces correct dtype and value through DSP")
    void testCastInt32ToInt64CorrectDtype() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        SDVariable x = sd.placeHolder("x", DataType.INT32, 1);
        SDVariable casted = sd.castTo("casted", x, DataType.INT64);
        // Add identity downstream to force external output
        SDVariable result = casted.add("result", sd.constant("zero", Nd4j.createFromArray(new long[]{0})));

        INDArray input = Nd4j.createFromArray(new int[]{42});

        Map<String, INDArray> expected = sd.output(Map.of("x", input), "result");
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "result");

        assertEquals(DataType.INT64, expected.get("result").dataType(),
                "Expected output should be INT64");
        assertEquals(42L, dspResult.get("result").getLong(0),
                "DSP cast result should be 42 (not garbage from dtype mismatch)");

        sd.close();
    }

    /**
     * Test cast to INT64 followed by strided_slice with multiple step values.
     * Runs multiple iterations to verify the Triton-compiled kernel consistently
     * produces correct results, not just on the first execution.
     */
    @Test
    @DisplayName("Cast INT64 → strided_slice: multi-step correctness")
    void testCastToStridedSliceMultiStep() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        SDVariable data = sd.placeHolder("data", DataType.FLOAT, 1, 10);
        SDVariable endIdx = sd.placeHolder("endIdx", DataType.INT32, 1);
        SDVariable endInt64 = sd.castTo("endInt64", endIdx, DataType.INT64);

        SDVariable begin = sd.constant("begin", Nd4j.createFromArray(new long[]{0, 0}));
        SDVariable strides = sd.constant("strides", Nd4j.createFromArray(new long[]{1, 1}));
        SDVariable one = sd.constant("one", Nd4j.createFromArray(new long[]{1}));
        SDVariable endArr = sd.concat("endArr", 0, one, endInt64);

        SDVariable sliced = sd.stridedSlice("sliced", data, begin, endArr, strides);
        SDVariable result = sd.sum("result", sliced);

        INDArray dataArr = Nd4j.linspace(1, 10, 10, DataType.FLOAT).reshape(1, 10);

        // Test with different end indices
        int[] endValues = {3, 5, 7, 10, 1, 8};
        for (int endVal : endValues) {
            INDArray endIdxArr = Nd4j.createFromArray(new int[]{endVal});

            Map<String, INDArray> expected = sd.output(
                    Map.of("data", dataArr, "endIdx", endIdxArr), "result");
            Map<String, INDArray> dspResult = sd.outputDirect(
                    Map.of("data", dataArr, "endIdx", endIdxArr), "result");

            double expectedSum = expected.get("result").getDouble(0);
            double dspSum = dspResult.get("result").getDouble(0);

            log.info("endIdx={}: expected={}, dsp={}", endVal, expectedSum, dspSum);
            assertEquals(expectedSum, dspSum, TOL,
                    "DSP result should match for endIdx=" + endVal +
                    " (if garbage, cast dtype was wrong → FLOAT32 bits read as INT64)");
        }

        sd.close();
    }

    /**
     * Test that cast FLOAT32→INT64 (not just INT32→INT64) works correctly.
     * This tests a different cast path where the source type is float.
     */
    @Test
    @DisplayName("Cast FLOAT32→INT64 produces correct value through DSP")
    void testCastFloat32ToInt64() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1);
        SDVariable casted = sd.castTo("casted", x, DataType.INT64);
        SDVariable result = casted.add("result", sd.constant("zero", Nd4j.createFromArray(new long[]{0})));

        INDArray input = Nd4j.createFromArray(new float[]{7.0f});

        Map<String, INDArray> expected = sd.output(Map.of("x", input), "result");
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "result");

        assertEquals(7L, dspResult.get("result").getLong(0),
                "DSP cast FLOAT32→INT64 should produce 7 (not garbage)");

        sd.close();
    }

    /**
     * Test cast to INT32 (not just INT64) with downstream view-capable op.
     * Ensures the fix handles all integer target types, not just INT64.
     */
    @Test
    @DisplayName("Cast FLOAT32→INT32 produces correct value through DSP")
    void testCastFloat32ToInt32() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1);
        SDVariable casted = sd.castTo("casted", x, DataType.INT32);
        SDVariable result = casted.add("result", sd.constant("zero", Nd4j.createFromArray(new int[]{0})));

        INDArray input = Nd4j.createFromArray(new float[]{42.0f});

        Map<String, INDArray> expected = sd.output(Map.of("x", input), "result");
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "result");

        assertEquals(42, dspResult.get("result").getInt(0),
                "DSP cast FLOAT32→INT32 should produce 42 (not garbage)");

        sd.close();
    }
}
