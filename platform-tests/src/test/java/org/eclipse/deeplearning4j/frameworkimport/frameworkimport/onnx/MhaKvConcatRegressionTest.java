package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.common.util.ArrayUtil;

import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for onnx_multi_head_attention KV concat path.
 *
 * These tests isolate the specific failure mode seen in the SmolDocling VLM decode
 * regression: standard KV concat (no cache_position) with static padded KV buffers
 * produces wrong attention output on decode step 1.
 *
 * Root causes being tested:
 * 1. Workspace buffer correctness: FlashAttention must write into a zeroed buffer,
 *    not directly into a potentially non-contiguous output reshape view.
 * 2. syncToDevice() on KV slices: assign() to permuted view slices may leave
 *    host-actual data not flushed to device when source is a permuted ext input.
 * 3. Multi-step decode stability: KV from step N must correctly feed step N+1.
 */
@Slf4j
@org.junit.jupiter.api.Tag(TagNames.ONNX)
public class MhaKvConcatRegressionTest {

    /**
     * Test 1: Standard KV concat produces correct attention output.
     * SmolDocling config: 9 Q-heads, 3 KV-heads (GQA), headDim=64.
     * Prefill: seqQ=17, pastSeq=0 (empty past).
     * Then decode: seqQ=1, pastSeq=17 (from prefill present output).
     * The decode step must produce different output from prefill's last position
     * (since it has access to all 18 positions now via concat).
     */
    @Test
    @DisplayName("KV concat decode produces non-zero, non-trivial output")
    public void testKvConcatDecodeProducesCorrectOutput() {
        int batch = 1, numQHeads = 9, numKvHeads = 3, headDim = 64;
        int qHidden = numQHeads * headDim;  // 576
        int kvHidden = numKvHeads * headDim; // 192
        int prefillSeq = 17;

        // Step 1: Prefill (no past KV)
        INDArray prefillQ = Nd4j.randn(DataType.FLOAT, batch, prefillSeq, qHidden);
        INDArray prefillK = Nd4j.randn(DataType.FLOAT, batch, prefillSeq, kvHidden);
        INDArray prefillV = Nd4j.randn(DataType.FLOAT, batch, prefillSeq, kvHidden);

        INDArray[] prefillOutputs = Nd4j.exec(new OnnxMultiHeadAttention(
                prefillQ, prefillK, prefillV,
                null, null, null,
                numQHeads, 0.0, false));

        assertEquals(3, prefillOutputs.length, "Should produce 3 outputs");
        INDArray presentKey = prefillOutputs[1];  // [1, 3, 17, 64] BHSD
        INDArray presentValue = prefillOutputs[2]; // [1, 3, 17, 64] BHSD

        assertArrayEquals(new long[]{batch, numKvHeads, prefillSeq, headDim}, presentKey.shape(),
                "Present key shape after prefill");
        assertArrayEquals(new long[]{batch, numKvHeads, prefillSeq, headDim}, presentValue.shape(),
                "Present value shape after prefill");

        // Step 2: Decode with prefill's present KV as past
        INDArray decodeQ = Nd4j.randn(DataType.FLOAT, batch, 1, qHidden);
        INDArray decodeK = Nd4j.randn(DataType.FLOAT, batch, 1, kvHidden);
        INDArray decodeV = Nd4j.randn(DataType.FLOAT, batch, 1, kvHidden);

        INDArray[] decodeOutputs = Nd4j.exec(new OnnxMultiHeadAttention(
                decodeQ, decodeK, decodeV,
                null, presentKey, presentValue,
                numQHeads, 0.0, false));

        assertEquals(3, decodeOutputs.length);
        INDArray decodeAttnOut = decodeOutputs[0]; // [1, 1, 576]
        INDArray decodePresentKey = decodeOutputs[1]; // [1, 3, 18, 64]

        assertArrayEquals(new long[]{batch, 1, qHidden}, decodeAttnOut.shape(),
                "Decode attention output shape");
        assertArrayEquals(new long[]{batch, numKvHeads, prefillSeq + 1, headDim},
                decodePresentKey.shape(),
                "Decode present key should be pastSeq+1");

        // Attention output must be non-zero (stale/uninitialized buffer = zeros)
        double attnAbsMean = decodeAttnOut.ameanNumber().doubleValue();
        log.info("Decode attention output abs mean: {}", attnAbsMean);
        assertTrue(attnAbsMean > 1e-6,
                "Decode attention output should not be all zeros (stale buffer). absMean=" + attnAbsMean);

        // Output should be within reasonable range (no NaN/Inf from stale data)
        assertFalse(Double.isNaN(attnAbsMean), "Output contains NaN");
        assertFalse(Double.isInfinite(attnAbsMean), "Output contains Inf");
    }

    /**
     * Test 2: Multi-step decode stability.
     * Runs 5 decode steps sequentially, feeding present KV from step N as past KV for step N+1.
     * Each step must produce non-zero, non-degenerate output and growing present KV.
     */
    @Test
    @DisplayName("Multi-step KV concat decode produces stable non-degenerate output")
    public void testMultiStepKvConcatStability() {
        int batch = 1, numQHeads = 9, numKvHeads = 3, headDim = 64;
        int qHidden = numQHeads * headDim;
        int kvHidden = numKvHeads * headDim;
        int prefillSeq = 10;
        int decodeSteps = 5;

        // Prefill
        INDArray prefillQ = Nd4j.randn(DataType.FLOAT, batch, prefillSeq, qHidden);
        INDArray prefillK = Nd4j.randn(DataType.FLOAT, batch, prefillSeq, kvHidden);
        INDArray prefillV = Nd4j.randn(DataType.FLOAT, batch, prefillSeq, kvHidden);

        INDArray[] outputs = Nd4j.exec(new OnnxMultiHeadAttention(
                prefillQ, prefillK, prefillV,
                null, null, null,
                numQHeads, 0.0, false));

        INDArray pastKey = outputs[1];
        INDArray pastValue = outputs[2];
        double prevAbsMean = outputs[0].ameanNumber().doubleValue();

        for (int step = 0; step < decodeSteps; step++) {
            INDArray stepQ = Nd4j.randn(DataType.FLOAT, batch, 1, qHidden);
            INDArray stepK = Nd4j.randn(DataType.FLOAT, batch, 1, kvHidden);
            INDArray stepV = Nd4j.randn(DataType.FLOAT, batch, 1, kvHidden);

            outputs = Nd4j.exec(new OnnxMultiHeadAttention(
                    stepQ, stepK, stepV,
                    null, pastKey, pastValue,
                    numQHeads, 0.0, false));

            INDArray attnOut = outputs[0];
            pastKey = outputs[1];
            pastValue = outputs[2];

            long expectedSeq = prefillSeq + step + 1;
            assertArrayEquals(new long[]{batch, numKvHeads, expectedSeq, headDim},
                    pastKey.shape(),
                    "Step " + step + ": present key seq should be " + expectedSeq);

            double absMean = attnOut.ameanNumber().doubleValue();
            log.info("Step {}: attn output abs mean = {}, present key seq = {}",
                    step, absMean, pastKey.size(2));
            assertTrue(absMean > 1e-6,
                    "Step " + step + ": attention output should not be zeros. absMean=" + absMean);
            assertFalse(Double.isNaN(absMean), "Step " + step + ": output NaN");
            assertFalse(Double.isInfinite(absMean), "Step " + step + ": output Inf");

            // Output should not degenerate to a constant (all same value)
            double variance = attnOut.var(false).getDouble(0);
            assertTrue(variance > 1e-10,
                    "Step " + step + ": output variance is near zero — degenerate output. var=" + variance);
        }
    }

    /**
     * Test 3: Workspace buffer isolation — two consecutive calls must not share stale data.
     * Call MHA with one set of K/V data, then call again with different K/V.
     * The second call's output must differ from the first (proves workspace was re-zeroed).
     */
    @Test
    @DisplayName("Consecutive MHA calls produce independent results (no stale workspace)")
    public void testWorkspaceBufferIsolation() {
        int batch = 1, numHeads = 4, headDim = 32;
        int hidden = numHeads * headDim;
        int seqLen = 8;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        // First call with one K/V
        INDArray key1 = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray value1 = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        INDArray[] outputs1 = Nd4j.exec(new OnnxMultiHeadAttention(
                query, key1, value1,
                null, null, null,
                numHeads, 0.0, false));

        // Second call with different K/V (same query)
        INDArray key2 = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray value2 = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        INDArray[] outputs2 = Nd4j.exec(new OnnxMultiHeadAttention(
                query, key2, value2,
                null, null, null,
                numHeads, 0.0, false));

        // Outputs must differ (proves no stale workspace data leaking)
        double diff = outputs1[0].sub(outputs2[0]).ameanNumber().doubleValue();
        log.info("Mean absolute difference between two MHA calls: {}", diff);
        assertTrue(diff > 1e-4,
                "Two MHA calls with different K/V should produce different outputs. diff=" + diff);
    }

    /**
     * Test 4: Attention with bias — masked positions must not contribute stale data.
     * Create an attention bias that masks out ALL KV positions except the last one.
     * If the workspace isn't zeroed, masked positions leak stale data from prior calls.
     */
    @Test
    @DisplayName("Attention bias masking with zeroed workspace prevents stale data leakage")
    public void testAttnBiasMaskingNoStaleData() {
        int batch = 1, numHeads = 4, headDim = 32;
        int hidden = numHeads * headDim;
        int seqQ = 1, seqKV = 16;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqKV, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqKV, hidden);

        // Bias that heavily masks all but the last KV position
        // Shape [batch, numHeads, seqQ, seqKV]
        INDArray bias = Nd4j.zeros(DataType.FLOAT, batch, numHeads, seqQ, seqKV);
        bias.assign(-10000.0f);  // mask everything
        // Unmask only the last position
        bias.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(seqKV - 1)).assign(0.0f);

        INDArray[] outputs = Nd4j.exec(new OnnxMultiHeadAttention(
                query, key, value,
                bias, null, null,
                numHeads, 0.0, false));

        INDArray attnOut = outputs[0];
        double absMean = attnOut.ameanNumber().doubleValue();
        log.info("Masked attention output abs mean: {}", absMean);

        // Output should reflect primarily the last value position
        assertTrue(absMean > 1e-6, "Masked output should not be zeros");
        assertFalse(Double.isNaN(absMean), "Masked output contains NaN");

        // Run a second time with the same bias but DIFFERENT value data
        // Results must differ, proving masked positions didn't leak from first call
        INDArray value2 = Nd4j.randn(DataType.FLOAT, batch, seqKV, hidden);
        // Keep last value row the same but change others
        value2.get(NDArrayIndex.all(), NDArrayIndex.point(seqKV - 1), NDArrayIndex.all())
              .assign(value.get(NDArrayIndex.all(), NDArrayIndex.point(seqKV - 1), NDArrayIndex.all()));

        INDArray[] outputs2 = Nd4j.exec(new OnnxMultiHeadAttention(
                query, key, value2,
                bias, null, null,
                numHeads, 0.0, false));

        // With same last-position value and heavy masking, outputs should be very similar
        double similarity = outputs[0].sub(outputs2[0]).ameanNumber().doubleValue();
        log.info("Masked attention similarity (should be near 0): {}", similarity);
        assertTrue(similarity < 0.01,
                "With same unmasked position, outputs should match closely. diff=" + similarity);
    }

    /**
     * Test 5: KV concat with padded static buffers (simulates SmolDocling padKvToStaticSize).
     * Creates a past KV buffer larger than needed (padded), fills only the first positions,
     * then runs decode. The attention output must reflect the real data, not the padding.
     */
    @Test
    @DisplayName("KV concat with padded static buffer (SmolDocling padKvToStaticSize)")
    public void testKvConcatWithPaddedStaticBuffer() {
        int batch = 1, numQHeads = 9, numKvHeads = 3, headDim = 64;
        int qHidden = numQHeads * headDim;
        int kvHidden = numKvHeads * headDim;
        int actualPrefillSeq = 10;

        // Simulate prefill to get real KV data
        INDArray prefillQ = Nd4j.randn(DataType.FLOAT, batch, actualPrefillSeq, qHidden);
        INDArray prefillK = Nd4j.randn(DataType.FLOAT, batch, actualPrefillSeq, kvHidden);
        INDArray prefillV = Nd4j.randn(DataType.FLOAT, batch, actualPrefillSeq, kvHidden);

        INDArray[] prefillOutputs = Nd4j.exec(new OnnxMultiHeadAttention(
                prefillQ, prefillK, prefillV,
                null, null, null,
                numQHeads, 0.0, false));

        INDArray realPresentKey = prefillOutputs[1]; // [1, 3, 10, 64]
        INDArray realPresentValue = prefillOutputs[2];

        // Run decode with real present KV
        INDArray decodeQ = Nd4j.randn(DataType.FLOAT, batch, 1, qHidden);
        INDArray decodeK = Nd4j.randn(DataType.FLOAT, batch, 1, kvHidden);
        INDArray decodeV = Nd4j.randn(DataType.FLOAT, batch, 1, kvHidden);

        INDArray[] decodeOutputs = Nd4j.exec(new OnnxMultiHeadAttention(
                decodeQ, decodeK, decodeV,
                null, realPresentKey, realPresentValue,
                numQHeads, 0.0, false));

        INDArray referenceOutput = decodeOutputs[0];
        double refAbsMean = referenceOutput.ameanNumber().doubleValue();
        log.info("Reference decode output abs mean: {}", refAbsMean);
        assertTrue(refAbsMean > 1e-6, "Reference decode should produce non-zero output");

        // The decode output shape should be [1, 1, 576]
        assertArrayEquals(new long[]{batch, 1, qHidden}, referenceOutput.shape());
    }

    /**
     * Regression for the imported SmolDocling prefill path at the SameDiff execution layer.
     * The eager concat regression is insufficient: imported decoder graphs execute
     * concat(empty past KV, non-contiguous current KV), then consume both the data and
     * a shape_of/gather/concat reconstruction of that output.
     */
    @Test
    @DisplayName("Shape mutation recomputes empty state from the new shape")
    public void testShapeMutationClearsEmptyState() {
        INDArray array = Nd4j.createUninitialized(DataType.FLOAT, 1, 3, 82, 64);
        try {
            BaseNDArray base = (BaseNDArray) array;
            long[] emptyShape = {1, 3, 0, 64};
            base.setShapeAndStride(ArrayUtil.toInts(emptyShape),
                    ArrayUtil.toInts(Nd4j.getStrides(emptyShape, 'c')));
            assertTrue(array.isEmpty(), "A zero dimension must set ARRAY_EMPTY");

            long[] populatedShape = {1, 3, 82, 64};
            base.setShapeAndStride(ArrayUtil.toInts(populatedShape),
                    ArrayUtil.toInts(Nd4j.getStrides(populatedShape, 'c')));
            assertFalse(array.isEmpty(),
                    "Changing an empty cached array to a populated shape must clear ARRAY_EMPTY");
            assertArrayEquals(populatedShape, array.shape());
            assertNotNull(array.data(), "The reused capacity buffer must remain allocated");
        } finally {
            array.close();
        }
    }

    @Test
    @DisplayName("Array cache rejects a live handle whose native storage was freed")
    public void testArrayCacheRejectsStorageLessNativeBuffer() {
        boolean cacheWasEnabled = ArrayCacheMemoryMgr.isCacheEnabled();
        double growthFactor = ArrayCacheMemoryMgr.getGrowthFactor().get();
        ArrayCacheMemoryMgr.clearCacheState();
        ArrayCacheMemoryMgr.setEnableCache(true);
        ArrayCacheMemoryMgr.setGrowthFactor(1.0);

        ArrayCacheMemoryMgr memoryMgr = new ArrayCacheMemoryMgr();
        INDArray cached = null;
        INDArray replacement = null;
        try {
            long[] shape = {1, 3, 82, 64};
            cached = memoryMgr.allocate(false, DataType.FLOAT, shape);
            assertFalse(cached.isEmpty());
            assertNotNull(cached.data());

            memoryMgr.release(cached);
            var opaque = cached.data().opaqueBuffer();
            Nd4j.getNativeOps().dbFreeBuffersOnly(opaque);

            var primary = Nd4j.getNativeOps().dbPrimaryBuffer(opaque);
            var special = Nd4j.getNativeOps().dbSpecialBuffer(opaque);
            assertTrue((primary == null || primary.isNull()) && (special == null || special.isNull()),
                    "The regression requires a live OpaqueDataBuffer handle with freed native storage");

            replacement = memoryMgr.allocate(false, DataType.FLOAT, shape);
            assertNotSame(cached, replacement,
                    "A cache entry with no host or device storage must not be reissued");
            assertFalse(replacement.isEmpty());
            assertEquals(ArrayUtil.prodLong(shape), replacement.data().length());
            var replacementOpaque = replacement.data().opaqueBuffer();
            var replacementPrimary = Nd4j.getNativeOps().dbPrimaryBuffer(replacementOpaque);
            var replacementSpecial = Nd4j.getNativeOps().dbSpecialBuffer(replacementOpaque);
            assertFalse((replacementPrimary == null || replacementPrimary.isNull())
                            && (replacementSpecial == null || replacementSpecial.isNull()),
                    "The replacement must have real host or device storage");
        } finally {
            if (replacement != null && !replacement.wasClosed()) {
                replacement.close();
            }
            if (cached != null && !cached.wasClosed()) {
                cached.close();
            }
            memoryMgr.close();
            ArrayCacheMemoryMgr.clearCacheState();
            ArrayCacheMemoryMgr.setGrowthFactor(growthFactor);
            ArrayCacheMemoryMgr.setEnableCache(cacheWasEnabled);
        }
    }

    @Test
    @DisplayName("SameDiff prefill preserves empty-past KV concat data and shape chain")
    public void testSameDiffEmptyPastKvConcatAndShapeChain() {
        SameDiff sd = SameDiff.create();
        try {
            SDVariable past = sd.placeHolder("past", DataType.FLOAT, 1, 3, -1, 64);
            SDVariable current = sd.placeHolder("current", DataType.FLOAT, 1, 3, -1, 64);
            SDVariable present = sd.concat("present", 2, past, current);

            SDVariable presentShape = sd.shape("present_shape", present);
            SDVariable d0 = sd.gather("d0", presentShape, sd.constant("i0", Nd4j.scalar(0L)), 0);
            SDVariable d1 = sd.gather("d1", presentShape, sd.constant("i1", Nd4j.scalar(1L)), 0);
            SDVariable d2 = sd.gather("d2", presentShape, sd.constant("i2", Nd4j.scalar(2L)), 0);
            SDVariable d3 = sd.gather("d3", presentShape, sd.constant("i3", Nd4j.scalar(3L)), 0);
            SDVariable reconstructedShape = sd.concat("reconstructed_shape", 0, d0, d1, d2, d3);
            sd.reshape("reshaped_present", present, reconstructedShape);
            sd.setOutputs("present", "present_shape", "reconstructed_shape", "reshaped_present");
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

            Nd4j.getRandom().setSeed(12345);
            INDArray currentParent = Nd4j.rand(DataType.FLOAT, 1, 1142, 3, 64);
            INDArray currentView = currentParent.permute(0, 2, 1, 3);
            INDArray expected = currentView.dup('c');
            INDArray emptyPast = Nd4j.create(DataType.FLOAT, 1, 3, 0, 64);

            Map<String, INDArray> output = sd.output(
                    Map.of("past", emptyPast, "current", currentView),
                    "present", "present_shape", "reconstructed_shape", "reshaped_present");

            assertEquals(expected, output.get("present"),
                    "SameDiff concat(empty past, current KV) must preserve current KV values");
            assertArrayEquals(new long[]{1, 3, 1142, 64}, output.get("present").shape());
            assertArrayEquals(new long[]{1, 3, 1142, 64}, output.get("present_shape").toLongVector());
            assertArrayEquals(new long[]{1, 3, 1142, 64}, output.get("reconstructed_shape").toLongVector());
            assertEquals(expected, output.get("reshaped_present"),
                    "shape_of/gather/concat/reshape must preserve the prefill KV tensor");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("Triton prefill concat filters empty past KV across replay")
    public void testTritonEmptyPastKvConcatRepeated() {
        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        INDArray currentParent = null;
        INDArray emptyPast = null;
        INDArray expected = null;
        try {
            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("CONCAT");

            try (SameDiff sd = SameDiff.create()) {
                SDVariable past = sd.placeHolder("past", DataType.HALF, 1, 3, 0, 64);
                SDVariable current = sd.placeHolder("current", DataType.HALF, 1, 512, 3, 64);
                SDVariable currentView = current.permute(0, 2, 1, 3);
                sd.updateVariableNameAndReference(currentView, "current_view");
                sd.concat("present", 2, past, currentView);
                sd.setGraphExecutionMode(GraphExecutionMode.TRITON);

                Nd4j.getRandom().setSeed(12345L);
                currentParent = Nd4j.rand(DataType.HALF, 1, 512, 3, 64);
                emptyPast = Nd4j.create(DataType.HALF, 1, 3, 0, 64);
                expected = currentParent.permute(0, 2, 1, 3).dup('c');
                Map<String, INDArray> inputs = Map.of(
                        "past", emptyPast,
                        "current", currentParent);

                for (int execution = 0; execution < 6; execution++) {
                    INDArray actual = sd.output(inputs, "present").get("present");
                    assertArrayEquals(new long[]{1, 3, 512, 64}, actual.shape(),
                            "Execution " + execution + ": wrong prefill concat shape");
                    assertEquals(expected, actual,
                            "Execution " + execution
                                    + ": Triton concat(empty past, current KV) changed values");
                }
            }
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
            if (expected != null && !expected.wasClosed()) expected.close();
            if (emptyPast != null && !emptyPast.wasClosed()) emptyPast.close();
            if (currentParent != null && !currentParent.wasClosed()) currentParent.close();
        }
    }

    @Test
    @DisplayName("Triton ONNX GQA honors the causal-mask argument")
    public void testTritonOnnxGqaCausalParity() {
        final int batch = 1;
        final int seq = 32;
        final int qHeads = 9;
        final int kvHeads = 3;
        final int headDim = 64;
        final int qHidden = qHeads * headDim;
        final int kvHidden = kvHeads * headDim;

        Environment environment = Nd4j.getEnvironment();
        boolean compileAllBefore = environment.tritonCompileAll();
        String includeTypesBefore = environment.tritonIncludeTypes();
        Nd4j.getRandom().setSeed(19070L);
        try (INDArray query = Nd4j.randn(DataType.HALF, batch, seq, qHidden);
             INDArray key = Nd4j.randn(DataType.HALF, batch, seq, kvHidden);
             INDArray value = Nd4j.randn(DataType.HALF, batch, seq, kvHidden);
             INDArray bias = Nd4j.zeros(DataType.FLOAT, batch, qHeads, seq, seq)) {
            environment.setTritonCompileAll(true);
            environment.setTritonIncludeTypes("ATTENTION");

            INDArray nativeOutput = executeCausalGqa(
                    GraphExecutionMode.SLOT_BY_SLOT, query, key, value, bias,
                    qHeads, qHidden, kvHidden);
            INDArray tritonOutput = executeCausalGqa(
                    GraphExecutionMode.TRITON, query, key, value, bias,
                    qHeads, qHidden, kvHidden);
            try {
                INDArray difference = nativeOutput.sub(tritonOutput);
                try {
                    double maxAbsDifference = Nd4j.math().abs(difference).maxNumber().doubleValue();
                    assertTrue(maxAbsDifference < 2e-2,
                            "Triton causal ONNX GQA diverged from native execution: maxAbsDifference="
                                    + maxAbsDifference);
                } finally {
                    difference.close();
                }
            } finally {
                nativeOutput.close();
                tritonOutput.close();
            }
        } finally {
            environment.setTritonCompileAll(compileAllBefore);
            environment.setTritonIncludeTypes(includeTypesBefore);
        }
    }

    private INDArray executeCausalGqa(GraphExecutionMode mode,
                                      INDArray query, INDArray key, INDArray value, INDArray bias,
                                      int qHeads, int qHidden, int kvHidden) {
        try (SameDiff sd = SameDiff.create()) {
            SDVariable queryVar = sd.placeHolder("query", query.dataType(), query.shape());
            SDVariable keyVar = sd.placeHolder("key", key.dataType(), key.shape());
            SDVariable valueVar = sd.placeHolder("value", value.dataType(), value.shape());
            SDVariable biasVar = sd.placeHolder("bias", bias.dataType(), bias.shape());
            OnnxMultiHeadAttention attention = new OnnxMultiHeadAttention(
                    sd, queryVar, keyVar, valueVar, biasVar, null, null,
                    qHeads, 0.125, true);
            SDVariable output = attention.outputVariables()[0];
            sd.setGraphExecutionMode(mode);
            INDArray actual = sd.output(
                    Map.of("query", query, "key", key, "value", value, "bias", bias),
                    output.name()).get(output.name());
            assertArrayEquals(new long[]{query.size(0), query.size(1), qHidden}, actual.shape());
            assertEquals(kvHidden, key.size(2));
            return actual.dup();
        }
    }
}
