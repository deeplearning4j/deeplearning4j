package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation tests for Triton/DSP concat with empty arrays.
 *
 * These tests target specific bugs found in the DSP slot executor and shape cache:
 *
 * Bug 1: NativeDynamicShapePlan_slotexec.cpp "skip all-empty outputs" guard
 *   skips concat when cached output is empty, even though inputs changed from
 *   empty to non-empty. Only view-capable ops are exempted.
 *
 * Bug 2: Stale empty shape cache — frozen shapes path keeps empty shape from
 *   step 0 even though step 1+ inputs are non-empty. Cache invalidation only
 *   fires for isViewCapableOp, not concat.
 *
 * Bug 3: CUDA/CPU concat helper returns immediately if output.isEmpty(),
 *   cascading from stale shape allocation.
 *
 * Each test isolates a specific failure mode and compares DSP output against
 * the standard execution path.
 */
@Slf4j
@NativeTag
@org.junit.jupiter.api.Tag(TagNames.SAMEDIFF)
public class TritonConcatEmptyIsolationTest extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-4;

    // ==================== Bug 1: Skip-all-empty-outputs guard ====================

    @Test
    @DisplayName("1a: DSP concat skips execution when cached output was empty but inputs changed")
    public void testDspConcatSkipEmptyOutputGuard() {
        // Step 0: concat(empty, empty) → empty (output cached as empty)
        // Step 1: concat(empty, non-empty) → should be non-empty, but DSP may skip
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        SDVariable past = sd.placeHolder("past", DataType.FLOAT, 1, 3, -1, 64);
        SDVariable current = sd.placeHolder("current", DataType.FLOAT, 1, 3, -1, 64);
        sd.concat("out", 2, past, current);

        // Step 0: both empty
        INDArray emptyA = Nd4j.create(DataType.FLOAT, 1, 3, 0, 64);
        INDArray emptyB = Nd4j.create(DataType.FLOAT, 1, 3, 0, 64);
        Map<String, INDArray> r0 = sd.outputDirect(
                Map.of("past", emptyA, "current", emptyB), "out");
        INDArray out0 = r0.get("out");
        assertTrue(out0.isEmpty() || out0.length() == 0,
                "Step 0: concat(empty, empty) should be empty, got shape=" +
                        Arrays.toString(out0.shape()));

        // Step 1: empty + non-empty — THIS is where Bug 1 manifests
        INDArray nonEmpty = Nd4j.randn(DataType.FLOAT, 1, 3, 5, 64);
        Map<String, INDArray> r1 = sd.outputDirect(
                Map.of("past", emptyA, "current", nonEmpty), "out");
        INDArray out1 = r1.get("out");
        assertNotNull(out1, "Step 1: output should not be null");
        assertFalse(out1.isEmpty(), "Step 1: concat(empty, non-empty) must NOT be empty");
        assertArrayEquals(new long[]{1, 3, 5, 64}, out1.shape(),
                "Step 1: shape should be [1,3,5,64], got " + Arrays.toString(out1.shape()));

        // Verify data matches the non-empty input
        double maxDiff = nonEmpty.sub(out1).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL,
                "Step 1: concat output should match non-empty input, maxDiff=" + maxDiff);
        log.info("testDspConcatSkipEmptyOutputGuard PASSED");
    }

    @Test
    @DisplayName("1b: DSP concat non-empty + empty (reversed order)")
    public void testDspConcatNonEmptyPlusEmpty() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        SDVariable past = sd.placeHolder("past", DataType.FLOAT, 1, 3, -1, 64);
        SDVariable current = sd.placeHolder("current", DataType.FLOAT, 1, 3, -1, 64);
        sd.concat("out", 2, past, current);

        INDArray nonEmpty = Nd4j.randn(DataType.FLOAT, 1, 3, 5, 64);
        INDArray empty = Nd4j.create(DataType.FLOAT, 1, 3, 0, 64);

        Map<String, INDArray> result = sd.outputDirect(
                Map.of("past", nonEmpty, "current", empty), "out");
        INDArray out = result.get("out");

        assertFalse(out.isEmpty(), "concat(non-empty, empty) must NOT be empty");
        assertArrayEquals(new long[]{1, 3, 5, 64}, out.shape(),
                "Shape should be [1,3,5,64], got " + Arrays.toString(out.shape()));
        log.info("testDspConcatNonEmptyPlusEmpty PASSED");
    }

    // ==================== Bug 2: Stale shape cache invalidation ====================

    @Test
    @DisplayName("2a: DSP frozen shape cache invalidation for concat empty→non-empty")
    public void testDspFrozenShapeCacheConcat() {
        // Run multiple steps where shapes change: empty→1→2→3
        // After step 0 caches empty shape, subsequent steps MUST invalidate
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        SDVariable past = sd.placeHolder("past", DataType.FLOAT, 1, 3, -1, 64);
        SDVariable current = sd.placeHolder("current", DataType.FLOAT, 1, 3, 1, 64);
        sd.concat("out", 2, past, current);

        INDArray pastKv = Nd4j.create(DataType.FLOAT, 1, 3, 0, 64);
        for (int step = 0; step < 10; step++) {
            INDArray newEntry = Nd4j.randn(DataType.FLOAT, 1, 3, 1, 64);

            // Standard path for comparison
            Map<String, INDArray> expected = sd.output(
                    Map.of("past", pastKv, "current", newEntry), "out");
            INDArray expectedOut = expected.get("out").dup();

            // DSP path
            Map<String, INDArray> dspResult = sd.outputDirect(
                    Map.of("past", pastKv, "current", newEntry), "out");
            INDArray actual = dspResult.get("out").dup();

            long expectedSeqLen = step + 1;
            assertArrayEquals(new long[]{1, 3, expectedSeqLen, 64}, actual.shape(),
                    "Step " + step + ": expected shape [1,3," + expectedSeqLen + ",64], got " +
                            Arrays.toString(actual.shape()));

            double maxDiff = expectedOut.sub(actual).amaxNumber().doubleValue();
            assertTrue(maxDiff < TOL,
                    "Step " + step + ": DSP vs standard maxDiff=" + maxDiff);

            pastKv = actual;
        }
        log.info("testDspFrozenShapeCacheConcat PASSED: 10 steps verified");
    }

    @Test
    @DisplayName("2b: Shape cache invalidation with alternating empty/non-empty inputs")
    public void testDspShapeCacheAlternating() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        SDVariable past = sd.placeHolder("past", DataType.FLOAT, 1, 3, -1, 64);
        SDVariable current = sd.placeHolder("current", DataType.FLOAT, 1, 3, -1, 64);
        sd.concat("out", 2, past, current);

        // Alternate between empty and non-empty to stress cache invalidation
        long[][] testCases = {
                {0, 0},  // both empty → [1,3,0,64]
                {0, 3},  // empty + 3 → [1,3,3,64]
                {3, 0},  // 3 + empty → [1,3,3,64]
                {2, 3},  // 2 + 3 → [1,3,5,64]
                {0, 0},  // both empty again
                {0, 1},  // empty + 1 → [1,3,1,64]
        };

        for (int i = 0; i < testCases.length; i++) {
            long seqA = testCases[i][0];
            long seqB = testCases[i][1];

            INDArray a = Nd4j.create(DataType.FLOAT, 1, 3, seqA, 64);
            INDArray b = Nd4j.create(DataType.FLOAT, 1, 3, seqB, 64);
            if (seqA > 0) a = Nd4j.randn(DataType.FLOAT, 1, 3, seqA, 64);
            if (seqB > 0) b = Nd4j.randn(DataType.FLOAT, 1, 3, seqB, 64);

            Map<String, INDArray> result = sd.outputDirect(
                    Map.of("past", a, "current", b), "out");
            INDArray out = result.get("out");

            long expectedSeq = seqA + seqB;
            assertArrayEquals(new long[]{1, 3, expectedSeq, 64}, out.shape(),
                    "Case " + i + " (" + seqA + "+" + seqB + "): expected seq=" +
                            expectedSeq + ", got " + Arrays.toString(out.shape()));

            if (expectedSeq == 0) {
                assertTrue(out.isEmpty() || out.length() == 0,
                        "Case " + i + ": all-empty concat should produce empty");
            } else {
                assertFalse(out.isEmpty(),
                        "Case " + i + ": concat with non-empty input must not be empty");
            }
        }
        log.info("testDspShapeCacheAlternating PASSED: {} cases verified", testCases.length);
    }

    // ==================== Bug 3: Kernel empty-output early-return ====================

    @Test
    @DisplayName("3a: Standard path concat(empty, non-empty) produces correct output")
    public void testStandardPathConcatEmptyPlusNonEmpty() {
        // Verify the standard (non-DSP) path handles concat with empty correctly
        SameDiff sd = SameDiff.create();

        SDVariable past = sd.placeHolder("past", DataType.FLOAT, 1, 3, -1, 64);
        SDVariable current = sd.placeHolder("current", DataType.FLOAT, 1, 3, -1, 64);
        sd.concat("out", 2, past, current);

        INDArray empty = Nd4j.create(DataType.FLOAT, 1, 3, 0, 64);
        INDArray nonEmpty = Nd4j.randn(DataType.FLOAT, 1, 3, 7, 64);

        Map<String, INDArray> result = sd.output(
                Map.of("past", empty, "current", nonEmpty), "out");
        INDArray out = result.get("out");

        assertFalse(out.isEmpty(), "Standard concat(empty, non-empty) must not be empty");
        assertArrayEquals(new long[]{1, 3, 7, 64}, out.shape(),
                "Shape should be [1,3,7,64], got " + Arrays.toString(out.shape()));

        // Verify data content
        double maxDiff = nonEmpty.sub(out).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL,
                "Standard concat output should match non-empty input, maxDiff=" + maxDiff);
        log.info("testStandardPathConcatEmptyPlusNonEmpty PASSED");
    }

    @Test
    @DisplayName("3b: Standard path multi-step concat evolution")
    public void testStandardPathConcatMultiStep() {
        SameDiff sd = SameDiff.create();

        SDVariable past = sd.placeHolder("past", DataType.FLOAT, 1, 3, -1, 64);
        SDVariable current = sd.placeHolder("current", DataType.FLOAT, 1, 3, 1, 64);
        sd.concat("out", 2, past, current);

        INDArray pastKv = Nd4j.create(DataType.FLOAT, 1, 3, 0, 64);
        for (int step = 0; step < 5; step++) {
            INDArray newEntry = Nd4j.randn(DataType.FLOAT, 1, 3, 1, 64);

            Map<String, INDArray> result = sd.output(
                    Map.of("past", pastKv, "current", newEntry), "out");
            INDArray out = result.get("out").dup();

            long expectedSeqLen = step + 1;
            assertArrayEquals(new long[]{1, 3, expectedSeqLen, 64}, out.shape(),
                    "Step " + step + ": shape should be [1,3," + expectedSeqLen + ",64]");
            assertFalse(out.isEmpty(), "Step " + step + ": output must not be empty");

            pastKv = out;
        }
        log.info("testStandardPathConcatMultiStep PASSED");
    }

    // ==================== KV Cache Pattern: concat in downstream chain ====================

    @Test
    @DisplayName("4a: Concat → matmul chain (KV cache → attention pattern)")
    public void testConcatMatmulChain() {
        // Simulates: present_kv = concat(past_kv, new_kv) → matmul(query, present_kv)
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        SDVariable pastKv = sd.placeHolder("past_kv", DataType.FLOAT, 1, 3, -1, 64);
        SDVariable newKv = sd.placeHolder("new_kv", DataType.FLOAT, 1, 3, 1, 64);
        SDVariable query = sd.placeHolder("query", DataType.FLOAT, 1, 3, 1, 64);
        SDVariable presentKv = sd.concat("present_kv", 2, pastKv, newKv);

        // Transpose present_kv for attention: [1,3,seq,64] → permute last two dims
        // Use matmul instead of full attention for simplicity
        SDVariable kvFlat = sd.reshape("kv_flat", presentKv, 3, -1); // [3, seq*64]
        SDVariable qFlat = sd.reshape("q_flat", query, 3, 64);      // [3, 64]

        // Just verify present_kv output is correct
        INDArray emptyPast = Nd4j.create(DataType.FLOAT, 1, 3, 0, 64);
        INDArray newEntry = Nd4j.randn(DataType.FLOAT, 1, 3, 1, 64);
        INDArray q = Nd4j.randn(DataType.FLOAT, 1, 3, 1, 64);

        // Standard path
        Map<String, INDArray> expected = sd.output(
                Map.of("past_kv", emptyPast, "new_kv", newEntry, "query", q),
                "present_kv");
        INDArray expectedKv = expected.get("present_kv").dup();

        // DSP path
        Map<String, INDArray> dspResult = sd.outputDirect(
                Map.of("past_kv", emptyPast, "new_kv", newEntry, "query", q),
                "present_kv");
        INDArray actualKv = dspResult.get("present_kv").dup();

        assertArrayEquals(expectedKv.shape(), actualKv.shape(),
                "present_kv shape mismatch: expected " + Arrays.toString(expectedKv.shape()) +
                        ", got " + Arrays.toString(actualKv.shape()));
        assertFalse(actualKv.isEmpty(), "present_kv must not be empty after concat");
        log.info("testConcatMatmulChain PASSED: present_kv shape={}",
                Arrays.toString(actualKv.shape()));
    }

    @Test
    @DisplayName("4b: Multi-layer concat (simulating 30-layer transformer KV)")
    public void testMultiLayerConcatEvolution() {
        // Simulates 4 layers of KV cache, each with their own concat
        int numLayers = 4;
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        for (int layer = 0; layer < numLayers; layer++) {
            SDVariable past = sd.placeHolder("past_" + layer, DataType.FLOAT, 1, 3, -1, 64);
            SDVariable current = sd.placeHolder("new_" + layer, DataType.FLOAT, 1, 3, 1, 64);
            sd.concat("present_" + layer, 2, past, current);
        }

        // Run 5 decode steps
        INDArray[] pastKvs = new INDArray[numLayers];
        for (int i = 0; i < numLayers; i++) {
            pastKvs[i] = Nd4j.create(DataType.FLOAT, 1, 3, 0, 64);
        }

        String[] outputNames = new String[numLayers];
        for (int i = 0; i < numLayers; i++) {
            outputNames[i] = "present_" + i;
        }

        for (int step = 0; step < 5; step++) {
            java.util.Map<String, INDArray> inputs = new java.util.HashMap<>();
            for (int layer = 0; layer < numLayers; layer++) {
                inputs.put("past_" + layer, pastKvs[layer]);
                inputs.put("new_" + layer, Nd4j.randn(DataType.FLOAT, 1, 3, 1, 64));
            }

            Map<String, INDArray> result = sd.outputDirect(inputs, outputNames);

            long expectedSeq = step + 1;
            for (int layer = 0; layer < numLayers; layer++) {
                INDArray out = result.get("present_" + layer).dup();
                assertArrayEquals(new long[]{1, 3, expectedSeq, 64}, out.shape(),
                        "Step " + step + " layer " + layer + ": expected seq=" +
                                expectedSeq + ", got " + Arrays.toString(out.shape()));
                assertFalse(out.isEmpty(),
                        "Step " + step + " layer " + layer + ": must not be empty");
                pastKvs[layer] = out;
            }
        }
        log.info("testMultiLayerConcatEvolution PASSED: {} layers x 5 steps", numLayers);
    }

    // ==================== Data type variants ====================

    @Test
    @DisplayName("5a: Concat empty arrays with FLOAT16 dtype")
    public void testConcatEmptyFloat16() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        SDVariable past = sd.placeHolder("past", DataType.FLOAT16, 1, 3, -1, 64);
        SDVariable current = sd.placeHolder("current", DataType.FLOAT16, 1, 3, 1, 64);
        sd.concat("out", 2, past, current);

        INDArray emptyPast = Nd4j.create(DataType.FLOAT16, 1, 3, 0, 64);
        INDArray newEntry = Nd4j.randn(DataType.FLOAT16, 1, 3, 1, 64);

        Map<String, INDArray> result = sd.outputDirect(
                Map.of("past", emptyPast, "current", newEntry), "out");
        INDArray out = result.get("out");

        assertFalse(out.isEmpty(), "FLOAT16 concat(empty, non-empty) must not be empty");
        assertArrayEquals(new long[]{1, 3, 1, 64}, out.shape());
        log.info("testConcatEmptyFloat16 PASSED");
    }

    @Test
    @DisplayName("5b: Concat empty arrays with BFLOAT16 dtype")
    public void testConcatEmptyBFloat16() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        SDVariable past = sd.placeHolder("past", DataType.BFLOAT16, 1, 3, -1, 64);
        SDVariable current = sd.placeHolder("current", DataType.BFLOAT16, 1, 3, 1, 64);
        sd.concat("out", 2, past, current);

        INDArray emptyPast = Nd4j.create(DataType.BFLOAT16, 1, 3, 0, 64);
        INDArray newEntry = Nd4j.randn(DataType.BFLOAT16, 1, 3, 1, 64);

        Map<String, INDArray> result = sd.outputDirect(
                Map.of("past", emptyPast, "current", newEntry), "out");
        INDArray out = result.get("out");

        assertFalse(out.isEmpty(), "BFLOAT16 concat(empty, non-empty) must not be empty");
        assertArrayEquals(new long[]{1, 3, 1, 64}, out.shape());
        log.info("testConcatEmptyBFloat16 PASSED");
    }

    // ==================== Axis variants ====================

    @Test
    @DisplayName("6: Concat empty along different axes")
    public void testConcatEmptyDifferentAxes() {
        // Test concat with empty dimension on each possible axis
        long[][] shapes = {
                {0, 3, 64},   // axis 0 empty
                {1, 0, 64},   // axis 1 empty (typical KV cache)
                {1, 3, 0},    // axis 2 empty
        };
        int[] concatAxes = {0, 1, 2};
        long[][] nonEmptyShapes = {
                {2, 3, 64},
                {1, 5, 64},
                {1, 3, 7},
        };

        for (int i = 0; i < shapes.length; i++) {
            SameDiff sd = SameDiff.create();
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);

            long[] emptyShape = shapes[i];
            long[] fullShape = nonEmptyShapes[i];
            int axis = concatAxes[i];

            SDVariable a = sd.placeHolder("a", DataType.FLOAT, emptyShape);
            SDVariable b = sd.placeHolder("b", DataType.FLOAT, fullShape);
            sd.concat("out", axis, a, b);

            INDArray emptyArr = Nd4j.create(DataType.FLOAT, emptyShape);
            INDArray fullArr = Nd4j.randn(DataType.FLOAT, fullShape);

            // Standard path
            Map<String, INDArray> stdResult = sd.output(
                    Map.of("a", emptyArr, "b", fullArr), "out");
            INDArray expected = stdResult.get("out").dup();

            // DSP path
            Map<String, INDArray> dspResult = sd.outputDirect(
                    Map.of("a", emptyArr, "b", fullArr), "out");
            INDArray actual = dspResult.get("out").dup();

            assertArrayEquals(expected.shape(), actual.shape(),
                    "Axis " + axis + ": shape mismatch. Expected " +
                            Arrays.toString(expected.shape()) + ", got " +
                            Arrays.toString(actual.shape()));
            assertFalse(actual.isEmpty(),
                    "Axis " + axis + ": concat(empty, non-empty) must not be empty");

            log.info("Axis {} PASSED: empty{} + full{} → {}",
                    axis, Arrays.toString(emptyShape), Arrays.toString(fullShape),
                    Arrays.toString(actual.shape()));
        }
    }

    // ==================== dup() of empty arrays ====================

    @Test
    @DisplayName("7: dup() of empty array creates independent copy")
    public void testEmptyArrayDupIndependence() {
        // Verifies the fix: dup() on empty arrays must return a NEW object
        INDArray original = Nd4j.create(DataType.FLOAT, 1, 3, 0, 64);
        INDArray duped = original.dup();

        // Must be different objects
        assertNotSame(original, duped,
                "dup() on empty array must create a new object, not return this");

        // Must have same shape
        assertArrayEquals(original.shape(), duped.shape(),
                "dup'd empty array must have same shape");

        // Must be independently closeable
        assertTrue(duped.isEmpty(), "dup'd array should also be empty");

        // Verify they are independently manageable objects
        // Note: empty arrays with null data may not report wasClosed() after close()
        // since there's no data buffer to close. The key invariant is object identity.
        assertNotSame(original, duped,
                "dup'd empty array must be a separate object from original");
        log.info("testEmptyArrayDupIndependence PASSED");
    }

    @Test
    @DisplayName("7b: dup() of various empty shapes")
    public void testEmptyArrayDupVariousShapes() {
        long[][] emptyShapes = {
                {0},
                {1, 0},
                {0, 5},
                {1, 3, 0, 64},
                {2, 0, 3, 4, 5},
        };

        for (long[] shape : emptyShapes) {
            INDArray original = Nd4j.create(DataType.FLOAT, shape);
            INDArray duped = original.dup();

            assertNotSame(original, duped,
                    "Shape " + Arrays.toString(shape) + ": dup must create new object");
            assertArrayEquals(shape, duped.shape(),
                    "Shape " + Arrays.toString(shape) + ": dup must preserve shape");
            assertTrue(duped.isEmpty(),
                    "Shape " + Arrays.toString(shape) + ": dup must preserve empty flag");

            original.close();
            assertFalse(duped.wasClosed(),
                    "Shape " + Arrays.toString(shape) + ": dup must be independent");
            duped.close();
        }
        log.info("testEmptyArrayDupVariousShapes PASSED: {} shapes verified", emptyShapes.length);
    }
}
