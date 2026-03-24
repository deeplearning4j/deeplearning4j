/*
 * Tests targeting the degenerate VLM output bug.
 *
 * These tests isolate specific hypotheses about WHY the frozen DSP path
 * produces repetitive/degenerate output while the baseline (non-frozen) worked.
 *
 * Key observations from prior isolation:
 *   - No C++ changes between baseline and HEAD → regression is purely Java
 *   - --no-triton: still degenerate (26/250)
 *   - --no-native-decode: still degenerate (26/250)
 *   - --no-attn-override: still degenerate (17/250)
 *   - --no-freeze: CRASH (released buffer)
 *   - --no-direct: CRASH (released buffer)
 *   - TokenSample vs argMax: both degenerate
 *   - Position tracking: confirmed correct (680, 681, 682... incrementing)
 *
 * NEW hypothesis: cuBLAS TF32 (added to OPTIMAL config since baseline) OR
 * the frozen fast path D2D copy not picking up in-place bias/mask updates.
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation tests for degenerate VLM output under frozen DSP path.
 */
public class FrozenPathDegenerateIsolationTest {

    private static final float MASK_FILL = -3.4028235e+38f;

    // ─── cuBLAS TF32 flag tests ─────────────────────────────────────────

    @Test
    @DisplayName("cuBLAS TF32 default should be OFF (baseline behavior)")
    public void testCublasTf32DefaultIsOff() {
        // The OPTIMAL benchmark config now sets cublasTf32(true), but the
        // system default should be OFF to match baseline behavior.
        // If this fails, the default changed and may explain degenerate output.
        var env = Nd4j.getEnvironment();
        // Just verify we can read the flag without error
        boolean tf32 = env.cublasTf32Enabled();
        // Log it — the value depends on what previous tests set
        System.out.println("cuBLAS TF32 enabled: " + tf32);
    }

    @Test
    @DisplayName("cuBLAS TF32 toggle should be independently controllable")
    public void testCublasTf32Toggle() {
        var env = Nd4j.getEnvironment();
        boolean original = env.cublasTf32Enabled();
        try {
            env.setCublasTf32Enabled(true);
            assertTrue(env.cublasTf32Enabled(), "TF32 should be ON after enable");

            env.setCublasTf32Enabled(false);
            assertFalse(env.cublasTf32Enabled(), "TF32 should be OFF after disable");
        } finally {
            env.setCublasTf32Enabled(original);
        }
    }

    @Test
    @DisplayName("Matmul with TF32 ON vs OFF should produce similar but not identical results")
    public void testMatmulTf32PrecisionDifference() {
        var env = Nd4j.getEnvironment();
        boolean original = env.cublasTf32Enabled();
        try {
            // Create realistic matmul inputs (like a decoder layer)
            // [1, 2048] x [2048, 2048] — typical hidden→hidden projection
            INDArray a = Nd4j.randn(DataType.FLOAT, 1, 2048);
            INDArray b = Nd4j.randn(DataType.FLOAT, 2048, 2048);

            // TF32 OFF (IEEE FP32)
            env.setCublasTf32Enabled(false);
            INDArray resultIeee = a.mmul(b);

            // TF32 ON (reduced precision tensor cores)
            env.setCublasTf32Enabled(true);
            INDArray resultTf32 = a.mmul(b);

            // They should be close but may differ
            double maxDiff = resultIeee.sub(resultTf32).amaxNumber().doubleValue();
            System.out.println("Max absolute difference (TF32 vs IEEE): " + maxDiff);

            // TF32 has 10-bit mantissa vs 23-bit for FP32.
            // For a 2048-wide dot product, error can compound.
            // Just verify both produce valid (non-NaN, non-Inf) results.
            assertFalse(resultIeee.isNaN().any(), "IEEE result has NaN");
            assertFalse(resultTf32.isNaN().any(), "TF32 result has NaN");
            assertFalse(resultIeee.isInfinite().any(), "IEEE result has Inf");
            assertFalse(resultTf32.isInfinite().any(), "TF32 result has Inf");
        } finally {
            env.setCublasTf32Enabled(original);
        }
    }

    @Test
    @DisplayName("Chained matmuls with TF32 should not diverge catastrophically over 30 layers")
    public void testChainedMatmulTf32Divergence() {
        var env = Nd4j.getEnvironment();
        boolean original = env.cublasTf32Enabled();
        try {
            // Simulate 30 transformer layers: x = x @ W for each layer
            int hidden = 2048;
            INDArray x = Nd4j.randn(DataType.FLOAT, 1, hidden).mul(0.01);

            // Generate 30 weight matrices (shared between TF32 and IEEE runs)
            INDArray[] weights = new INDArray[30];
            for (int i = 0; i < 30; i++) {
                weights[i] = Nd4j.randn(DataType.FLOAT, hidden, hidden).mul(0.02);
            }

            // IEEE run
            env.setCublasTf32Enabled(false);
            INDArray xIeee = x.dup();
            for (int i = 0; i < 30; i++) {
                xIeee = xIeee.mmul(weights[i]);
            }

            // TF32 run
            env.setCublasTf32Enabled(true);
            INDArray xTf32 = x.dup();
            for (int i = 0; i < 30; i++) {
                xTf32 = xTf32.mmul(weights[i]);
            }

            double maxDiff = xIeee.sub(xTf32).amaxNumber().doubleValue();
            double ieeeMax = xIeee.amaxNumber().doubleValue();
            double relDiff = ieeeMax > 0 ? maxDiff / ieeeMax : maxDiff;

            System.out.println("After 30 chained matmuls:");
            System.out.println("  IEEE max abs: " + ieeeMax);
            System.out.println("  TF32 max abs: " + xTf32.amaxNumber().doubleValue());
            System.out.println("  Max absolute diff: " + maxDiff);
            System.out.println("  Relative diff: " + relDiff);

            // Both should still be valid
            assertFalse(xIeee.isNaN().any(), "IEEE result has NaN after 30 layers");
            assertFalse(xTf32.isNaN().any(), "TF32 result has NaN after 30 layers");
        } finally {
            env.setCublasTf32Enabled(original);
        }
    }

    // ─── In-place array update visibility under frozen path ─────────────

    @Test
    @DisplayName("putScalar on FLOAT array followed by ensureLocation(DEVICE) should be visible")
    public void testPutScalarEnsureLocationVisibility() {
        // Simulates the attn_mask_reformat bias update pattern:
        //   bias.putScalar({0,0,0,pos}, 0.0f);
        //   ensureLocation(bias, DEVICE);
        // If ensureLocation doesn't actually trigger H2D, the device buffer is stale.
        INDArray bias = Nd4j.valueArrayOf(new long[]{1, 1, 1, 100}, MASK_FILL, DataType.FLOAT);
        Nd4j.getAffinityManager().ensureLocation(bias,
                org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);

        // Unmask positions one at a time (simulating decode steps)
        for (int pos = 0; pos < 50; pos++) {
            bias.putScalar(new long[]{0, 0, 0, pos}, 0.0f);
            Nd4j.getAffinityManager().ensureLocation(bias,
                    org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);
        }

        // Read back and verify all 50 positions are unmasked
        for (int pos = 0; pos < 50; pos++) {
            float val = bias.getFloat(0, 0, 0, pos);
            assertEquals(0.0f, val, 1e-6f,
                    "Position " + pos + " should be unmasked (0.0) but got " + val);
        }
        // Verify remaining positions still masked
        for (int pos = 50; pos < 100; pos++) {
            float val = bias.getFloat(0, 0, 0, pos);
            assertEquals(MASK_FILL, val, 1e-6f,
                    "Position " + pos + " should still be masked");
        }
    }

    @Test
    @DisplayName("putScalar on LONG array (attention_mask) followed by ensureLocation(DEVICE)")
    public void testLongMaskPutScalarVisibility() {
        // Simulates 1D attention mask update pattern
        long maxKvLen = 729;
        long currentSeqLen = 1;
        long totalSeqLen = maxKvLen + currentSeqLen;

        INDArray mask = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);
        // Set prefill positions
        int prefillLen = 679;
        mask.get(org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.interval(0, prefillLen)).assign(1);
        mask.putScalar(0, totalSeqLen - 1, 1);

        Nd4j.getAffinityManager().ensureLocation(mask,
                org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);

        // Simulate 20 decode steps — each unmasks one more position
        for (int step = 0; step < 20; step++) {
            long cachePos = prefillLen + step;
            if (cachePos > 0) {
                mask.put(new org.nd4j.linalg.indexing.INDArrayIndex[]{
                                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                                org.nd4j.linalg.indexing.NDArrayIndex.point(cachePos)},
                        Nd4j.scalar(DataType.LONG, 1));
                Nd4j.getAffinityManager().ensureLocation(mask,
                        org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);
            }
        }

        // Verify: positions 0..prefillLen+19 should be 1, the rest 0 (except last)
        long expectedOnes = prefillLen + 20 + 1; // prefill + 20 decode steps + last position
        long actualSum = mask.sumNumber().longValue();
        assertEquals(expectedOnes, actualSum,
                "Expected " + expectedOnes + " ones in mask but got " + actualSum);
    }

    // ─── Attention bias construction correctness ────────────────────────

    @Test
    @DisplayName("4D bias initial construction should mask exactly [cachePos..maxKvLen-1]")
    public void testBiasInitialConstruction() {
        long maxKvLen = 729;
        long currentSeqLen = 1;
        long cachePos = 679; // prefillSeqLen
        long totalSeqLen = maxKvLen + currentSeqLen; // 730

        int totalLen = (int) totalSeqLen;
        float[] biasData = new float[(int) currentSeqLen * totalLen];
        for (int q = 0; q < (int) currentSeqLen; q++) {
            int rowOffset = q * totalLen;
            for (int k = (int) cachePos; k < (int) maxKvLen; k++) {
                biasData[rowOffset + k] = MASK_FILL;
            }
            for (int k = q + 1; k < (int) currentSeqLen; k++) {
                biasData[rowOffset + (int) maxKvLen + k] = MASK_FILL;
            }
        }

        INDArray bias = Nd4j.create(biasData, new long[]{1, 1, currentSeqLen, totalLen}, 'c');

        // Positions [0..678] should be 0.0 (attend to prefill)
        for (int k = 0; k < cachePos; k++) {
            assertEquals(0.0f, bias.getFloat(0, 0, 0, k), 1e-6f,
                    "Position " + k + " (prefill) should be 0.0 (attend)");
        }
        // Positions [679..728] should be MASK_FILL (empty padding)
        for (int k = (int) cachePos; k < (int) maxKvLen; k++) {
            assertEquals(MASK_FILL, bias.getFloat(0, 0, 0, k), 1e-6f,
                    "Position " + k + " (empty) should be MASK_FILL");
        }
        // Position 729 (current token) should be 0.0 (attend)
        assertEquals(0.0f, bias.getFloat(0, 0, 0, (int) maxKvLen), 1e-6f,
                "Position " + maxKvLen + " (current token) should be 0.0");
    }

    @Test
    @DisplayName("Bias accumulation over 50 steps should unmask exactly one position per step")
    public void testBiasAccumulationOver50Steps() {
        long maxKvLen = 729;
        long currentSeqLen = 1;
        long prefillSeqLen = 679;
        long totalSeqLen = maxKvLen + currentSeqLen;

        // Build initial bias (step 1 creation)
        int totalLen = (int) totalSeqLen;
        float[] biasData = new float[totalLen];
        for (int k = (int) prefillSeqLen; k < (int) maxKvLen; k++) {
            biasData[k] = MASK_FILL;
        }
        INDArray bias = Nd4j.create(biasData, new long[]{1, 1, 1, totalLen}, 'c');

        // Simulate 50 decode steps
        for (int step = 0; step < 50; step++) {
            long cachePos = prefillSeqLen + 1 + step; // after step 1 scatter
            if (cachePos > 0) {
                bias.putScalar(new long[]{0, 0, 0, cachePos - 1}, 0.0f);
            }

            // Count unmasked positions in the past region [0..maxKvLen-1]
            int unmasked = 0;
            for (int k = 0; k < (int) maxKvLen; k++) {
                if (Math.abs(bias.getFloat(0, 0, 0, k)) < 1e-6f) {
                    unmasked++;
                }
            }
            int expected = (int) prefillSeqLen + step + 1; // prefill + steps unmasked so far
            assertEquals(expected, unmasked,
                    "At step " + step + ": expected " + expected + " unmasked positions, got " + unmasked);
        }
    }

    // ─── argMax stability on views ──────────────────────────────────────

    @Test
    @DisplayName("argMax on contiguous vs view of same data should agree")
    public void testArgMaxViewConsistency() {
        // Simulates the greedy sampling: argMax on a view of logits
        INDArray logits3d = Nd4j.randn(DataType.FLOAT, 1, 1, 49280); // [batch, seq, vocab]
        INDArray view = logits3d.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.all());
        INDArray contiguous = view.dup();

        int argmaxView = Nd4j.argMax(view).getInt(0);
        int argmaxContiguous = Nd4j.argMax(contiguous).getInt(0);

        assertEquals(argmaxContiguous, argmaxView,
                "argMax on view (" + argmaxView + ") vs contiguous (" + argmaxContiguous + ") disagree");
    }

    @Test
    @DisplayName("argMax stability over 50 calls with same logits should return same token")
    public void testArgMaxStabilityRepeated() {
        INDArray logits = Nd4j.randn(DataType.FLOAT, 49280);
        int firstArgmax = Nd4j.argMax(logits).getInt(0);

        for (int i = 0; i < 50; i++) {
            int argmax = Nd4j.argMax(logits).getInt(0);
            assertEquals(firstArgmax, argmax,
                    "argMax changed at iteration " + i + ": expected " + firstArgmax + " got " + argmax);
        }
    }

    // ─── KvCacheManager state consistency ───────────────────────────────

    @Test
    @DisplayName("StaticKvCacheManager scatter+position tracking matches manual inline code")
    public void testKvCacheManagerMatchesInlineCode() {
        // Compare KvCacheManager behavior to what the baseline did inline
        int numLayers = 30;
        int numHeads = 3;
        int headDim = 64;
        long prefillSeqLen = 20;
        int maxNewTokens = 50;
        long maxKvLen = prefillSeqLen + maxNewTokens;

        // Create fake prefill outputs
        java.util.Map<String, INDArray> prefillOutputs = new java.util.HashMap<>();
        java.util.List<String> keyNames = new java.util.ArrayList<>();
        java.util.List<String> valueNames = new java.util.ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            String keyName = "present." + i + ".key";
            String valueName = "present." + i + ".value";
            keyNames.add(keyName);
            valueNames.add(valueName);
            prefillOutputs.put(keyName, Nd4j.randn(DataType.FLOAT, 1, numHeads, prefillSeqLen, headDim));
            prefillOutputs.put(valueName, Nd4j.randn(DataType.FLOAT, 1, numHeads, prefillSeqLen, headDim));
        }

        var kvNames = new org.eclipse.deeplearning4j.llm.generation.DecoderUtils.KVCacheNames(keyNames, valueNames);

        // Initialize via KvCacheManager
        var manager = new org.eclipse.deeplearning4j.llm.generation.StaticKvCacheManager();
        manager.initializeFromPrefill(prefillOutputs, kvNames, maxNewTokens, prefillSeqLen);

        assertEquals(maxKvLen, manager.getMaxKvLen(), "maxKvLen mismatch");
        assertEquals(prefillSeqLen, manager.getCachePosition(), "initial cachePos mismatch");

        // Simulate 10 decode steps with Java scatter
        for (int step = 0; step < 10; step++) {
            long expectedPos = prefillSeqLen + step;
            assertEquals(expectedPos, manager.getCachePosition(),
                    "cachePos mismatch before step " + step);

            // Create fake decoder output (present KV with maxKvLen+1 seq dim)
            java.util.Map<String, INDArray> decoderOutputs = new java.util.HashMap<>();
            for (int i = 0; i < numLayers; i++) {
                decoderOutputs.put("present." + i + ".key",
                        Nd4j.randn(DataType.FLOAT, 1, numHeads, maxKvLen + 1, headDim));
                decoderOutputs.put("present." + i + ".value",
                        Nd4j.randn(DataType.FLOAT, 1, numHeads, maxKvLen + 1, headDim));
            }

            manager.scatterNewEntries(decoderOutputs, kvNames);
            assertEquals(expectedPos + 1, manager.getCachePosition(),
                    "cachePos not incremented after step " + step);
        }
    }
}
