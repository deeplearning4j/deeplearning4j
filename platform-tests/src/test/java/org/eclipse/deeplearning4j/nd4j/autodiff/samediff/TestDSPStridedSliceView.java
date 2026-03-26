package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanCompiler;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAGBuilder;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@DisplayName("DSP StridedSlice View Op Tests")
public class TestDSPStridedSliceView {

    private static final double TOL = 1e-5;

    private DynamicShapePlan compilePlan(SameDiff sd, String... outputs) {
        Set<String> outSet = new LinkedHashSet<>();
        for (String o : outputs) outSet.add(o);
        ForwardExecutionDAGBuilder builder = new ForwardExecutionDAGBuilder(sd);
        var dag = builder.buildForwardDAG(outSet);
        return DynamicShapePlanCompiler.compile(sd, dag, outSet);
    }

    // ── Compiler flag tests ──────────────────────────────────────────────

    @Test
    @DisplayName("Compiler flags strided_slice as viewCapableOp")
    void testCompilerFlagsStridedSliceAsView() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        // slice rows [1:3] → shape [2,8]
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{1, 0}, new long[]{3, 8}, new long[]{1, 1});
        SDVariable out = sliced.mul("out", sd.constant("s", Nd4j.scalar(2.0f)));

        DynamicShapePlan plan = compilePlan(sd, "out");
        assertNotNull(plan, "Plan should compile");

        boolean found = false;
        for (DynamicShapeSlot slot : plan.getSlots()) {
            if ("strided_slice".equals(slot.getOpName())) {
                assertTrue(slot.isViewCapableOp(), "strided_slice must be flagged viewCapableOp");
                found = true;
            }
        }
        assertTrue(found, "strided_slice slot must exist in plan");
        sd.close();
    }

    @Test
    @DisplayName("Compiler: strided_slice with shrink_axis is still viewCapable")
    void testCompilerStridedSliceShrinkAxis() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        // shrink first dim at index 2 → shape [8]
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{2, 0}, new long[]{3, 8}, new long[]{1, 1},
                0, 0, 0, 0, 1); // shrinkAxisMask=1 (bit 0)
        SDVariable out = sliced.add("out", sd.constant("b", Nd4j.scalar(1.0f)));

        DynamicShapePlan plan = compilePlan(sd, "out");
        assertNotNull(plan);

        for (DynamicShapeSlot slot : plan.getSlots()) {
            if ("strided_slice".equals(slot.getOpName())) {
                assertTrue(slot.isViewCapableOp());
            }
        }
        sd.close();
    }

    // ── Correctness: standard vs DSP execution ──────────────────────────

    @Test
    @DisplayName("Basic slice [1:3,:] matches standard execution under DSP")
    void testBasicSliceDSPCorrectness() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{1, 0}, new long[]{3, 8}, new long[]{1, 1});
        SDVariable out = sliced.mul("out", sd.constant("s", Nd4j.scalar(3.0f)));

        INDArray input = Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(4, 8);

        // Standard execution
        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        // DSP execution
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        INDArray actual = sd.output(Map.of("x", input), "out").get("out");

        assertArrayEquals(expected.shape(), actual.shape(), "Shapes must match");
        double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, "Slice DSP result diverges by " + maxDiff);
        sd.close();
    }

    @Test
    @DisplayName("Slice with stride=2 matches standard execution under DSP")
    void testStridedSliceWithStrideDSP() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 6, 4);
        // Every other row: [0:6:2, :] → shape [3,4]
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{0, 0}, new long[]{6, 4}, new long[]{2, 1});
        SDVariable out = sliced.add("out", sd.constant("b", Nd4j.scalar(10.0f)));

        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(6, 4);

        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        INDArray actual = sd.output(Map.of("x", input), "out").get("out");

        assertArrayEquals(expected.shape(), actual.shape());
        double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, "Strided slice DSP result diverges by " + maxDiff);
        sd.close();
    }

    @Test
    @DisplayName("Slice with shrink_axis_mask matches standard execution under DSP")
    void testShrinkAxisDSPCorrectness() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        // Select row 2 → shape [8] (shrink dim 0)
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{2, 0}, new long[]{3, 8}, new long[]{1, 1},
                0, 0, 0, 0, 1);
        SDVariable out = sliced.mul("out", sd.constant("s", Nd4j.scalar(5.0f)));

        INDArray input = Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(4, 8);

        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        INDArray actual = sd.output(Map.of("x", input), "out").get("out");

        assertArrayEquals(expected.shape(), actual.shape(), "Shapes must match after shrink");
        double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, "Shrink axis DSP result diverges by " + maxDiff);
        sd.close();
    }

    @Test
    @DisplayName("Slice with begin_mask matches standard execution under DSP")
    void testBeginMaskDSPCorrectness() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 6, 4);
        // beginMask=1 (bit 0): ignore begin for dim 0, so [:3, :]
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{99, 0}, new long[]{3, 4}, new long[]{1, 1},
                1, 0, 0, 0, 0); // beginMask=1
        SDVariable out = sliced.add("out", sd.constant("b", Nd4j.scalar(1.0f)));

        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(6, 4);

        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        INDArray actual = sd.output(Map.of("x", input), "out").get("out");

        assertArrayEquals(expected.shape(), actual.shape());
        double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, "Begin mask DSP result diverges by " + maxDiff);
        sd.close();
    }

    @Test
    @DisplayName("Slice with end_mask matches standard execution under DSP")
    void testEndMaskDSPCorrectness() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 6, 4);
        // endMask=1 (bit 0): ignore end for dim 0, so [2:, :]
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{2, 0}, new long[]{99, 4}, new long[]{1, 1},
                0, 1, 0, 0, 0); // endMask=1
        SDVariable out = sliced.mul("out", sd.constant("s", Nd4j.scalar(2.0f)));

        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(6, 4);

        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        INDArray actual = sd.output(Map.of("x", input), "out").get("out");

        assertArrayEquals(expected.shape(), actual.shape());
        double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, "End mask DSP result diverges by " + maxDiff);
        sd.close();
    }

    // ── Multi-step execution (view data freshness) ──────────────────────

    @Test
    @DisplayName("Slice view reflects new input data across multiple steps")
    void testSliceViewMultiStep() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 4);
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{1, 0}, new long[]{3, 4}, new long[]{1, 1});
        SDVariable out = sliced.mul("out", sd.constant("s", Nd4j.scalar(1.0f)));

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);

        INDArray[] results = new INDArray[5];
        INDArray[] expecteds = new INDArray[5];

        for (int step = 0; step < 5; step++) {
            INDArray input = Nd4j.rand(DataType.FLOAT, 4, 4).mul(step + 1);

            // Standard path for expected
            sd.setDspAutoCompileEnabled(false);
            sd.clearDynamicShapePlanCache();
            sd.resetSession();
            expecteds[step] = sd.output(Map.of("x", input), "out").get("out").dup();

            // DSP path
            sd.setDspAutoCompileEnabled(true);
            sd.clearDynamicShapePlanCache();
            sd.resetSession();
            results[step] = sd.output(Map.of("x", input), "out").get("out").dup();
        }

        for (int step = 0; step < 5; step++) {
            assertArrayEquals(expecteds[step].shape(), results[step].shape(),
                    "Step " + step + " shape mismatch");
            double maxDiff = results[step].sub(expecteds[step]).amaxNumber().doubleValue();
            assertTrue(maxDiff < TOL,
                    "Step " + step + " DSP result diverges by " + maxDiff);
        }

        // Verify steps produce different values (not stale)
        for (int step = 1; step < 5; step++) {
            double diff = results[step].sub(results[0]).amaxNumber().doubleValue();
            assertTrue(diff > TOL,
                    "Step " + step + " should differ from step 0 (not stale data)");
        }
        sd.close();
    }

    // ── Slice → downstream op chains ────────────────────────────────────

    @Test
    @DisplayName("Slice → matmul chain matches standard under DSP")
    void testSliceMatmulChainDSP() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        // Slice [1:3, :] → [2, 8]
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{1, 0}, new long[]{3, 8}, new long[]{1, 1});
        SDVariable w = sd.constant("w", Nd4j.rand(DataType.FLOAT, 8, 3));
        SDVariable out = sd.mmul("out", sliced, w);

        INDArray input = Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(4, 8);

        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        INDArray actual = sd.output(Map.of("x", input), "out").get("out");

        assertArrayEquals(expected.shape(), actual.shape());
        double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-2, "Slice→matmul DSP result diverges by " + maxDiff);
        sd.close();
    }

    @Test
    @DisplayName("Slice → add → relu chain matches standard under DSP")
    void testSliceAddReluChainDSP() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 6, 4);
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{0, 1}, new long[]{6, 3}, new long[]{1, 1});
        SDVariable bias = sd.constant("bias", Nd4j.ones(DataType.FLOAT, 6, 2));
        SDVariable added = sliced.add("added", bias);
        SDVariable out = sd.nn.relu("out", added, 0);

        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(6, 4);

        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        INDArray actual = sd.output(Map.of("x", input), "out").get("out");

        assertArrayEquals(expected.shape(), actual.shape());
        double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, "Slice→add→relu DSP result diverges by " + maxDiff);
        sd.close();
    }

    // ── 3D tensor slice (common in attention mask processing) ───────────

    @Test
    @DisplayName("3D slice [0:1, :, 0:seqLen] matches standard under DSP")
    void testSlice3DDSP() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4, 8);
        // Slice [0:1, :, 2:6] → [1, 4, 4]
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{0, 0, 2}, new long[]{1, 4, 6}, new long[]{1, 1, 1});
        SDVariable out = sliced.mul("out", sd.constant("s", Nd4j.scalar(2.0f)));

        INDArray input = Nd4j.linspace(1, 64, 64, DataType.FLOAT).reshape(2, 4, 8);

        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        INDArray actual = sd.output(Map.of("x", input), "out").get("out");

        assertArrayEquals(expected.shape(), actual.shape());
        double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, "3D slice DSP result diverges by " + maxDiff);
        sd.close();
    }

    @Test
    @DisplayName("4D attention-mask-style slice matches standard under DSP")
    void testSlice4DAttentionMaskDSP() {
        SameDiff sd = SameDiff.create();
        // [batch, heads, seqLen, seqLen]
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4, 16, 16);
        // Slice to [1, 4, 8, 16] — first 8 positions of the key dimension
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{0, 0, 0, 0}, new long[]{1, 4, 8, 16}, new long[]{1, 1, 1, 1});
        SDVariable out = sliced.add("out", sd.constant("b", Nd4j.scalar(0.0f)));

        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 4, 16, 16);

        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        INDArray actual = sd.output(Map.of("x", input), "out").get("out");

        assertArrayEquals(expected.shape(), actual.shape());
        double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, "4D attention mask slice DSP diverges by " + maxDiff);
        sd.close();
    }

    // ── Direct op execution (no SameDiff graph) ─────────────────────────

    @Test
    @DisplayName("Non-contiguous view add op produces correct result via Nd4j.exec")
    void testNonContiguousViewAddOp() {
        // Isolation test: does a non-contiguous view work correctly when passed
        // to a C++ op through OpContext?
        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(6, 4);

        // Create a column slice view: [6,2] with strides [4,1] and offset 1
        // This selects columns 1,2 from the [6,4] input
        INDArray view = Nd4j.create(input.data(), new long[]{6, 2}, new long[]{4, 1}, 1, 'c');

        // Verify the view reads correct values
        log.info("View shape: {} strides: {} offset: {}", java.util.Arrays.toString(view.shape()),
                java.util.Arrays.toString(view.stride()), view.offset());
        log.info("View:\n{}", view);

        // Expected column slice values (columns 1 and 2)
        assertEquals(2.0f, view.getFloat(0, 0), 1e-5f, "view[0,0] should be input[0,1]=2");
        assertEquals(3.0f, view.getFloat(0, 1), 1e-5f, "view[0,1] should be input[0,2]=3");
        assertEquals(6.0f, view.getFloat(1, 0), 1e-5f, "view[1,0] should be input[1,1]=6");
        assertEquals(7.0f, view.getFloat(1, 1), 1e-5f, "view[1,1] should be input[1,2]=7");

        // Now test add op with the view as input
        INDArray bias = Nd4j.ones(DataType.FLOAT, 6, 2);
        INDArray result = view.add(bias);

        log.info("Result:\n{}", result);
        assertEquals(3.0f, result.getFloat(0, 0), 1e-5f, "result[0,0] should be 2+1=3");
        assertEquals(4.0f, result.getFloat(0, 1), 1e-5f, "result[0,1] should be 3+1=4");
        assertEquals(7.0f, result.getFloat(1, 0), 1e-5f, "result[1,0] should be 6+1=7");
        assertEquals(8.0f, result.getFloat(1, 1), 1e-5f, "result[1,1] should be 7+1=8");
    }

    @Test
    @DisplayName("Direct StridedSlice op produces correct view output")
    void testDirectStridedSliceView() {
        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(4, 6);

        // Slice [1:3, 2:5] → [2, 3]
        INDArray result = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.StridedSlice(
                input,
                new long[]{1, 2}, new long[]{3, 5}, new long[]{1, 1},
                0, 0, 0, 0, 0))[0];

        assertEquals(2, result.rank());
        assertArrayEquals(new long[]{2, 3}, result.shape());

        // Verify values: input[1,2]=9, input[1,3]=10, input[1,4]=11
        //                input[2,2]=15, input[2,3]=16, input[2,4]=17
        assertEquals(9.0f, result.getFloat(0, 0), 1e-5f);
        assertEquals(10.0f, result.getFloat(0, 1), 1e-5f);
        assertEquals(11.0f, result.getFloat(0, 2), 1e-5f);
        assertEquals(15.0f, result.getFloat(1, 0), 1e-5f);
        assertEquals(16.0f, result.getFloat(1, 1), 1e-5f);
        assertEquals(17.0f, result.getFloat(1, 2), 1e-5f);
    }

    // ── Bug reproduction: SDVariable begin/end/strides (iArgs capped to 5) ──

    @Test
    @DisplayName("StridedSlice with SDVariable begin/end/strides matches standard under DSP")
    void testStridedSliceWithTensorInputsDSP() {
        // This reproduces the real-world DSP bug: when begin/end/strides come as
        // SDVariable input tensors (not literal arrays), DSP caps iArgs to 5 (masks only).
        // configureFromArguments() then sets begin/end/strides to zero-length arrays,
        // and the old null-check missed them → view was the entire input (wrong).
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        SDVariable beginVar = sd.constant("begin", Nd4j.createFromArray(new long[]{1, 0}));
        SDVariable endVar = sd.constant("end", Nd4j.createFromArray(new long[]{3, 8}));
        SDVariable stridesVar = sd.constant("strides", Nd4j.createFromArray(new long[]{1, 1}));
        // 4-input constructor: data, begin, end, strides — this triggers the iArgs capping path
        SDVariable sliced = sd.stridedSlice("sliced", x, beginVar, endVar, stridesVar);
        SDVariable out = sliced.mul("out", sd.constant("s", Nd4j.scalar(2.0f)));

        INDArray input = Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(4, 8);

        // Standard execution
        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        // DSP execution
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        INDArray actual = sd.output(Map.of("x", input), "out").get("out");

        assertArrayEquals(expected.shape(), actual.shape(), "Shapes must match");
        double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL,
                "Tensor-input strided_slice DSP result diverges by " + maxDiff);
        sd.close();
    }

    @Test
    @DisplayName("StridedSlice with SDVariable inputs on transposed array under DSP")
    void testStridedSliceTransposedInputDSP() {
        // Reproduces the exact model scenario: strided_slice on a transposed (non-contiguous)
        // BOOL array with SDVariable begin/end/strides inputs
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.BOOL, 1, 512, 512);
        // Transpose dims 1,2 → strides become non-contiguous
        SDVariable transposed = sd.permute("transposed", x, 0, 2, 1);
        SDVariable beginVar = sd.constant("begin", Nd4j.createFromArray(new long[]{0, 0, 0}));
        SDVariable endVar = sd.constant("end", Nd4j.createFromArray(new long[]{1, 256, 512}));
        SDVariable stridesVar = sd.constant("strides", Nd4j.createFromArray(new long[]{1, 1, 1}));
        SDVariable sliced = sd.stridedSlice("sliced", transposed, beginVar, endVar, stridesVar);
        // Cast to float so we can do math
        SDVariable floatSlice = sliced.castTo("floatSlice", DataType.FLOAT);
        SDVariable out = floatSlice.add("out", sd.constant("b", Nd4j.scalar(0.0f)));

        INDArray input = Nd4j.ones(DataType.BOOL, 1, 512, 512);

        // Standard execution
        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        // DSP execution
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        INDArray actual = sd.output(Map.of("x", input), "out").get("out");

        assertArrayEquals(expected.shape(), actual.shape(), "Shapes must match for transposed input");
        double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL,
                "Transposed strided_slice DSP result diverges by " + maxDiff);
        sd.close();
    }

    @Test
    @DisplayName("Direct StridedSlice with stride=2 produces correct values")
    void testDirectStridedSliceWithStride() {
        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(6, 4);

        // Every other row: [0:6:2, :] → [3, 4]
        INDArray result = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.StridedSlice(
                input,
                new long[]{0, 0}, new long[]{6, 4}, new long[]{2, 1},
                0, 0, 0, 0, 0))[0];

        assertArrayEquals(new long[]{3, 4}, result.shape());
        // Row 0: [1,2,3,4], Row 2: [9,10,11,12], Row 4: [17,18,19,20]
        assertEquals(1.0f, result.getFloat(0, 0), 1e-5f);
        assertEquals(9.0f, result.getFloat(1, 0), 1e-5f);
        assertEquals(17.0f, result.getFloat(2, 0), 1e-5f);
    }
}
