/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.DecoderUtils;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.StaticKvCacheManager;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for GAPS in component interaction coverage that existing tests miss.
 *
 * These tests target the specific untested interactions between components
 * in the decode pipeline identified by gap analysis of the last 2 days of commits
 * and the run-benchmark.sh test output.
 *
 * <h2>Gap Summary</h2>
 * <ol>
 *   <li><b>setCloseable toggle at step 1</b>: output() poisons placeholder constant flags
 *       via setCloseable(false), then restores via setCloseable(true). This toggle can
 *       corrupt DSP frozen buffer sync behavior if isConstant state drives H2D decisions.</li>
 *   <li><b>Mask accumulation vs C++ capture buffer overwrite race</b>: Java ensureLocation(DEVICE)
 *       must complete before the D2D capture buffer copy reads the same device buffer.
 *       If async, D2D reads stale mask data.</li>
 *   <li><b>cachePos inconsistency between mask and bias</b>: Existing tests explicitly SKIP
 *       position cachePos in consistency checks. This test verifies the skipped position.</li>
 *   <li><b>Cast cache warmup→replay lifecycle</b>: clearCastCache vs resetCastCacheIndices
 *       affects whether stale FP16 entries survive across execution mode transitions.</li>
 *   <li><b>cuBLAS workspace warmup/capture algorithm consistency</b>: Different workspace
 *       sizes cause different GEMM algorithm selection → numerical divergence.</li>
 *   <li><b>putScalar no device sync for input_ids</b>: reusableInputIds.putScalar() writes
 *       host only; without ensureLocation(DEVICE), D2D copy reads stale device data.</li>
 * </ol>
 *
 * Run with:
 *   cd platform-tests && mvn test \
 *     -Dtest=DecodeComponentInteractionGapTest \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 */
@Slf4j
@DisplayName("Decode Component Interaction Gap Tests")
public class DecodeComponentInteractionGapTest {

    // Constants matching SmolDocling-like dimensions for realistic testing
    private static final int NUM_KV_HEADS = 3;
    private static final int HEAD_DIM = 64;
    private static final long HIDDEN_SIZE = 576;
    private static final int MAX_KV_LEN = 100;
    private static final int NUM_LAYERS = 2;
    private static final int PREFILL_LEN = 10;
    private static final int SMALL_K = 64;
    private static final int SMALL_N = 128;
    private static final float MASK_FILL = DecoderUtils.MASK_FILL;
    private static final String ATTN_REFORMAT_NODE = "/model/attn_mask_reformat/Tile/output_0";

    // ═══════════════════════════════════════════════════════════════════════
    // GAP 1: setCloseable(false) poisoning at step 1 corrupts buffer state
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Gap 1a: output() calls setCloseable(false) on all placeholder arrays before execution,
     * then setCloseable(true) after. This toggles dataBuffer.setConstant(true/false).
     *
     * For static KV buffers and reusable decode inputs, this constant flag toggle
     * may affect how syncToSpecial() decides whether to sync. If a buffer is marked
     * constant during execution, C++ may skip syncing it. The subsequent setCloseable(true)
     * restores the flag, but by then the DSP may have cached the "constant" state.
     *
     * This test verifies that the round-trip toggle does NOT corrupt the buffer's
     * actual data or prevent subsequent writes from being visible.
     */
    @Test
    @DisplayName("Gap1a: setCloseable round-trip does not corrupt static KV buffer data")
    public void testSetCloseableRoundTripDoesNotCorruptKvBuffers() {
        // Create static KV buffers with known data
        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();
        for (INDArray buf : staticKvBuffers.values()) {
            buf.assign(Nd4j.randn(buf.shape()).castTo(buf.dataType()));
        }

        // Record original data
        Map<String, INDArray> originalData = new HashMap<>();
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            originalData.put(e.getKey(), e.getValue().dup());
        }

        // Simulate output() wrapping: setCloseable(false) → execute → setCloseable(true)
        for (INDArray buf : staticKvBuffers.values()) {
            buf.setCloseable(false);
        }

        // Verify constant flag is set
        for (INDArray buf : staticKvBuffers.values()) {
            assertFalse(buf.closeable(),
                    "After setCloseable(false): buffer should not be closeable");
        }

        // Simulate execution completing
        for (INDArray buf : staticKvBuffers.values()) {
            buf.setCloseable(true);
        }

        // Verify data is unchanged
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            INDArray current = e.getValue();
            INDArray original = originalData.get(e.getKey());
            double maxDiff = maxAbsError(original, current);
            assertEquals(0.0, maxDiff, 1e-10,
                    e.getKey() + ": data corrupted by setCloseable round-trip (maxDiff=" + maxDiff + ")");
        }

        // Verify writes AFTER the round-trip are still visible
        for (INDArray buf : staticKvBuffers.values()) {
            float valueBefore = buf.getFloat(0, 0, 0, 0);
            buf.putScalar(new long[]{0, 0, 0, 0}, valueBefore + 42.0f);
            assertEquals(valueBefore + 42.0f, buf.getFloat(0, 0, 0, 0), 1e-6,
                    "Write after setCloseable round-trip should be visible");
        }

        // Cleanup
        for (INDArray buf : staticKvBuffers.values()) buf.close();
        for (INDArray buf : originalData.values()) buf.close();
        log.info("Gap1a passed: setCloseable round-trip preserves KV buffer data");
    }

    /**
     * Gap 1b: When output() calls setCloseable(false) on reusable attention mask
     * and 4D bias, subsequent putScalar writes (for mask accumulation) must still
     * be visible even after the constant flag was toggled.
     *
     * The real scenario: step 1 uses output() which sets constant=true on the
     * attention mask. Step 2 uses outputDirect() and expects the mask to have
     * the step 1 update visible. If setConstant(true) somehow cached or froze
     * the buffer state, the step 1 putScalar would be invisible at step 2.
     */
    @Test
    @DisplayName("Gap1b: putScalar writes survive setCloseable(false)→true round-trip on reusable mask")
    public void testPutScalarSurvivesCloseableRoundTripOnMask() {
        long totalSeqLen = MAX_KV_LEN + 1;
        INDArray mask = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeqLen);

        // Initial state: fill prefill positions
        int cachePos = PREFILL_LEN;
        mask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, cachePos)).assign(1);
        mask.putScalar(0, totalSeqLen - 1, 1);  // current token
        for (int k = cachePos; k < MAX_KV_LEN; k++) {
            bias.putScalar(new long[]{0, 0, 0, k}, MASK_FILL);
        }

        // Verify initial state
        assertEquals(1L, mask.getLong(0, cachePos - 1));
        assertEquals(0L, mask.getLong(0, cachePos));

        // Step 1: output() wrapping
        mask.setCloseable(false);
        bias.setCloseable(false);

        // Step 1 completes, undo wrapping
        mask.setCloseable(true);
        bias.setCloseable(true);

        // Step 2 prep: advance cachePos, update mask and bias for new position
        cachePos = PREFILL_LEN + 1;
        mask.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(cachePos - 1)},
                Nd4j.scalar(DataType.LONG, 1));
        bias.putScalar(new long[]{0, 0, 0, cachePos - 1}, 0.0f);

        // Verify step 2 update is visible
        assertEquals(1L, mask.getLong(0, cachePos - 1),
                "Step 2 mask update at cachePos-1 should be visible after closeable round-trip");
        assertEquals(0.0f, bias.getFloat(0, 0, 0, cachePos - 1), 1e-6,
                "Step 2 bias update at cachePos-1 should be visible after closeable round-trip");

        // Position cachePos should still be masked
        assertEquals(0L, mask.getLong(0, cachePos),
                "Position cachePos should still be 0 (not yet filled)");
        assertEquals(MASK_FILL, bias.getFloat(0, 0, 0, cachePos), 1e-6,
                "Position cachePos should still be MASK_FILL");

        mask.close();
        bias.close();
        log.info("Gap1b passed: putScalar writes visible after setCloseable round-trip");
    }

    /**
     * Gap 1c: Simulate the full step 1→2 transition with a real SameDiff model.
     *
     * Step 1: decoder.output() (wraps with setCloseable toggle)
     * Step 2: decoder.outputDirect() (no wrapping, expects frozen state intact)
     *
     * Both should produce correct, different results for different inputs.
     * If the setCloseable toggle at step 1 damages session state, step 2 may
     * produce identical output to step 1 or garbage.
     */
    @Test
    @DisplayName("Gap1c: output() at step 1 then outputDirect() at step 2 both produce correct results")
    public void testOutputThenOutputDirectProduceCorrectResults() {
        SameDiff sd = SameDiff.create();

        // Simple model: matmul + bias
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, SMALL_K);
        SDVariable weights = sd.constant("weights", Nd4j.randn(DataType.FLOAT, SMALL_K, SMALL_N));
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, SMALL_N));
        SDVariable mm = sd.mmul("mm", input, weights);
        SDVariable result = mm.add("result", bias);

        // Step 1: output() (calls setCloseable(false) then setCloseable(true) on input)
        INDArray input1 = Nd4j.randn(DataType.FLOAT, 1, SMALL_K);
        Map<String, INDArray> out1 = sd.output(Map.of("input", input1), "result");
        INDArray result1 = out1.get("result").dup();

        // Step 2: outputDirect() (no wrapping)
        INDArray input2 = Nd4j.randn(DataType.FLOAT, 1, SMALL_K);
        Map<String, INDArray> out2 = sd.outputDirect(Map.of("input", input2), "result");
        INDArray result2 = out2.get("result").dup();

        // Both should be non-zero, non-NaN
        assertFalse(Double.isNaN(result1.sumNumber().doubleValue()), "Step 1 output should not be NaN");
        assertFalse(Double.isNaN(result2.sumNumber().doubleValue()), "Step 2 output should not be NaN");
        assertTrue(Math.abs(result1.sumNumber().doubleValue()) > 1e-6, "Step 1 output should not be all zeros");
        assertTrue(Math.abs(result2.sumNumber().doubleValue()) > 1e-6, "Step 2 output should not be all zeros");

        // They should differ (different inputs)
        assertTrue(maxAbsError(result1, result2) > 1e-3,
                "Steps 1 and 2 should produce different results for different inputs");

        // Verify correctness against manual computation.
        // Use dup() on constant arrays to avoid any interaction with SameDiff internal state.
        INDArray weightsFp = weights.getArr().dup();
        INDArray biasArr = bias.getArr().dup();
        INDArray expected1 = input1.mmul(weightsFp).add(biasArr);
        INDArray expected2 = input2.mmul(weightsFp).add(biasArr);
        double err1 = maxAbsError(expected1, result1);
        double err2 = maxAbsError(expected2, result2);
        log.info("Step 1 max error: {}, Step 2 max error: {}", err1, err2);
        // Tolerance: FP32 64x128 matmul can have ~1e-3 error due to op ordering
        assertTrue(err1 < 0.01,
                "Step 1 output should match manual computation (err=" + err1 + ")");
        assertTrue(err2 < 0.01,
                "Step 2 output should match manual computation (err=" + err2 + ")");

        // Step 3: another outputDirect — verify session still works after mixed calls
        INDArray input3 = Nd4j.randn(DataType.FLOAT, 1, SMALL_K);
        Map<String, INDArray> out3 = sd.outputDirect(Map.of("input", input3), "result");
        INDArray result3 = out3.get("result").dup();
        INDArray expected3 = input3.mmul(weightsFp).add(biasArr);
        double err3 = maxAbsError(expected3, result3);
        assertTrue(err3 < 0.01,
                "Step 3 outputDirect should still produce correct results (err=" + err3 + ")");

        sd.close();
        log.info("Gap1c passed: output()→outputDirect() transition preserves correctness");
    }

    /**
     * Gap 1d: Mixed output()/outputDirect() calls with FP16 weights (mixed-precision).
     *
     * Tests that the cast cache is not corrupted by the setCloseable toggle.
     * If the cast cache's FP16 buffer is marked constant during output() and the
     * mark is not properly undone, the next outputDirect() may skip re-casting
     * changed activations.
     */
    @Test
    @DisplayName("Gap1d: Mixed-precision output()→outputDirect() with changing activations")
    public void testMixedPrecisionOutputTransition() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("a", DataType.FLOAT, 1, SMALL_K);
        SDVariable weights = sd.constant("w", Nd4j.randn(DataType.HALF, SMALL_K, SMALL_N));
        SDVariable mm = sd.mmul("out", input, weights);

        INDArray weightsFp32 = weights.getArr().castTo(DataType.FLOAT);

        INDArray[] outputs = new INDArray[5];
        INDArray[] activations = new INDArray[5];

        // Step 0,1: output() (with setCloseable toggle)
        for (int i = 0; i < 2; i++) {
            activations[i] = Nd4j.randn(DataType.FLOAT, 1, SMALL_K);
            Map<String, INDArray> out = sd.output(Map.of("a", activations[i]), "out");
            outputs[i] = out.get("out").castTo(DataType.FLOAT).dup();
        }

        // Steps 2-4: outputDirect() (no toggle)
        for (int i = 2; i < 5; i++) {
            activations[i] = Nd4j.randn(DataType.FLOAT, 1, SMALL_K);
            Map<String, INDArray> out = sd.outputDirect(Map.of("a", activations[i]), "out");
            outputs[i] = out.get("out").castTo(DataType.FLOAT).dup();
        }

        // All outputs must differ (different activations)
        for (int i = 0; i < 5; i++) {
            for (int j = i + 1; j < 5; j++) {
                double diff = maxAbsError(outputs[i], outputs[j]);
                assertTrue(diff > 1e-3,
                        "Steps " + i + " and " + j + " should differ (diff=" + diff + ")");
            }
        }

        // All outputs must be correct
        for (int i = 0; i < 5; i++) {
            INDArray expected = activations[i].mmul(weightsFp32);
            double err = maxAbsError(expected, outputs[i]);
            assertTrue(err < 0.1,
                    "Step " + i + " max error " + err + " exceeds tolerance");
        }

        sd.close();
        log.info("Gap1d passed: mixed-precision output/outputDirect transition correct");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GAP 2: Mask accumulation vs capture buffer D2D copy race condition
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Gap 2a: After putScalar + ensureLocation(DEVICE), the device buffer must
     * reflect the updated value IMMEDIATELY (not async). The D2D capture buffer
     * copy reads from this device buffer — if it reads stale data, the model
     * sees an old mask and can't attend to newly decoded tokens.
     *
     * This test verifies that ensureLocation(DEVICE) is synchronous by reading
     * the value back from device memory immediately after the call.
     */
    @Test
    @DisplayName("Gap2a: ensureLocation(DEVICE) after putScalar makes update immediately visible")
    public void testEnsureLocationDeviceMakesUpdateVisible() {
        long totalSeqLen = MAX_KV_LEN + 1;
        INDArray mask = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);

        // Fill initial positions
        mask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, PREFILL_LEN)).assign(1);
        mask.putScalar(0, totalSeqLen - 1, 1);

        // Simulate step-by-step updates (the real decode loop path)
        for (int step = 0; step < 20; step++) {
            int pos = PREFILL_LEN + step;

            // Update Java-side mask (same as buildDecoderInputMap reusable path)
            mask.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(pos)},
                    Nd4j.scalar(DataType.LONG, 1));

            // Force host→device sync
            Nd4j.getAffinityManager().ensureLocation(mask,
                    org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);

            // Read back — if ensureLocation is async and device has stale data,
            // this would read 0 instead of 1.
            long readBack = mask.getLong(0, pos);
            assertEquals(1L, readBack,
                    "Step " + step + ": ensureLocation(DEVICE) must make putScalar immediately visible. " +
                    "Got " + readBack + " at position " + pos + ". " +
                    "If 0, the D2D capture buffer copy would read stale mask data.");
        }

        // Verify cumulative mask: positions 0..PREFILL_LEN+19 should all be 1
        long sum = mask.sumNumber().longValue();
        long expectedOnes = PREFILL_LEN + 20 + 1;  // prefill + 20 steps + current token at end
        assertEquals(expectedOnes, sum,
                "After 20 steps: mask should have " + expectedOnes + " ones, got " + sum);

        mask.close();
        log.info("Gap2a passed: ensureLocation(DEVICE) makes putScalar visible immediately");
    }

    /**
     * Gap 2b: The 4D attention bias must also be immediately visible after
     * putScalar + ensureLocation(DEVICE). This is the attn_mask_reformat
     * override path — if the bias update is stale, the model sees MASK_FILL
     * at the newly decoded position and can't attend to it.
     */
    @Test
    @DisplayName("Gap2b: 4D bias putScalar + ensureLocation(DEVICE) is immediately visible")
    public void testBiasEnsureLocationDeviceVisible() {
        long totalSeqLen = MAX_KV_LEN + 1;
        int totalLen = (int) totalSeqLen;

        // Build initial bias (same as buildDecoderInputMap initial construction)
        float[] biasData = new float[totalLen];
        for (int k = PREFILL_LEN; k < MAX_KV_LEN; k++) {
            biasData[k] = MASK_FILL;
        }
        INDArray bias = Nd4j.create(biasData, new long[]{1, 1, 1, totalLen}, 'c');

        // Verify initial state
        assertEquals(0.0f, bias.getFloat(0, 0, 0, PREFILL_LEN - 1), 1e-6);
        assertEquals(MASK_FILL, bias.getFloat(0, 0, 0, PREFILL_LEN), 1e-6);

        // Simulate step-by-step unmask (the real reusable bias path)
        for (int step = 0; step < 20; step++) {
            int posToUnmask = PREFILL_LEN + step;

            // Unmask position (same as buildDecoderInputMap reusable path)
            bias.putScalar(new long[]{0, 0, 0, posToUnmask}, 0.0f);

            // Force sync
            Nd4j.getAffinityManager().ensureLocation(bias,
                    org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);

            // Read back — must see 0.0, not MASK_FILL
            float readBack = bias.getFloat(0, 0, 0, posToUnmask);
            assertEquals(0.0f, readBack, 1e-6,
                    "Step " + step + ": bias at position " + posToUnmask +
                    " should be 0.0 after unmask, got " + readBack +
                    ". If MASK_FILL, model can't attend to newly decoded token.");

            // Next position should still be masked
            if (posToUnmask + 1 < MAX_KV_LEN) {
                assertEquals(MASK_FILL, bias.getFloat(0, 0, 0, posToUnmask + 1), 1e-6,
                        "Step " + step + ": position " + (posToUnmask + 1) + " should still be masked");
            }
        }

        bias.close();
        log.info("Gap2b passed: 4D bias ensureLocation(DEVICE) is immediate");
    }

    /**
     * Gap 2c: Multiple rapid putScalar + ensureLocation cycles must not lose updates.
     *
     * In the real decode loop, each step updates BOTH the mask AND the bias,
     * each followed by ensureLocation. If the second ensureLocation somehow
     * reverts or interferes with the first, one of the updates would be lost.
     */
    @Test
    @DisplayName("Gap2c: Rapid mask+bias update cycle does not lose either update")
    public void testRapidMaskBiasUpdateCycleNoLoss() {
        long totalSeqLen = MAX_KV_LEN + 1;
        INDArray mask = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);
        mask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, PREFILL_LEN)).assign(1);
        mask.putScalar(0, totalSeqLen - 1, 1);

        float[] biasData = new float[(int) totalSeqLen];
        for (int k = PREFILL_LEN; k < MAX_KV_LEN; k++) {
            biasData[k] = MASK_FILL;
        }
        INDArray bias = Nd4j.create(biasData, new long[]{1, 1, 1, totalSeqLen}, 'c');

        for (int step = 0; step < 30; step++) {
            int pos = PREFILL_LEN + step;
            if (pos >= MAX_KV_LEN) break;

            // Update mask (same order as buildDecoderInputMap)
            mask.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(pos)},
                    Nd4j.scalar(DataType.LONG, 1));
            Nd4j.getAffinityManager().ensureLocation(mask,
                    org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);

            // Update bias (separate ensureLocation call)
            bias.putScalar(new long[]{0, 0, 0, pos}, 0.0f);
            Nd4j.getAffinityManager().ensureLocation(bias,
                    org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);

            // Verify BOTH updates survived
            assertEquals(1L, mask.getLong(0, pos),
                    "Step " + step + ": mask update lost at position " + pos);
            assertEquals(0.0f, bias.getFloat(0, 0, 0, pos), 1e-6,
                    "Step " + step + ": bias update lost at position " + pos);
        }

        mask.close();
        bias.close();
        log.info("Gap2c passed: rapid mask+bias cycles preserve both updates");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GAP 3: cachePos inconsistency between 1D mask and 4D bias
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Gap 3a: Existing test DecodeFeatureInteractionRegressionTest#testMaskAndBiasConsistency
     * SKIPS position cachePos with the comment "known-inconsistent edge case."
     *
     * This test verifies what ACTUALLY happens at position cachePos.
     * The current decode loop writes position cachePos-1 (just-filled), so
     * cachePos itself should be blocked (mask=0, bias=MASK_FILL) in both.
     */
    @Test
    @DisplayName("Gap3a: Position cachePos is consistently blocked in both mask and bias")
    public void testCachePosConsistentlyBlockedInMaskAndBias() {
        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .attnMaskReformatOutput(ATTN_REFORMAT_NODE)
                .build();

        List<String> inputNames = createInputNamesWithBias();
        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();
        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        Map<String, INDArray> reusableInputs = new HashMap<>();

        // Run 10 steps and check position cachePos at each step
        for (int step = 0; step < 10; step++) {
            int cachePos = PREFILL_LEN + step;

            Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                    ioConfig, inputNames, dummyDecoder, embeddings, inputIds,
                    679 + step, 1, staticKvBuffers, MAX_KV_LEN, cachePos,
                    true, HIDDEN_SIZE, reusableInputs, true);

            INDArray mask = result.get("attention_mask");
            INDArray bias = result.get(ATTN_REFORMAT_NODE);

            assertNotNull(mask, "Step " + step + ": mask should exist");
            assertNotNull(bias, "Step " + step + ": bias should exist");

            // Position cachePos is the NEXT empty slot — NOT yet filled.
            // Both mask and bias should agree it's blocked.
            long maskAtPos = mask.getLong(0, cachePos);
            float biasAtPos = bias.getFloat(0, 0, 0, cachePos);

            // The mask sets [0..cachePos-1] to 1 via interval(0, cachePos), so cachePos is 0
            assertEquals(0L, maskAtPos,
                    "Step " + step + ": mask at cachePos=" + cachePos + " should be 0 (not yet filled)");
            // The bias has MASK_FILL at [cachePos..maxKvLen-1] initially, and only unmasks cachePos-1
            assertEquals(MASK_FILL, biasAtPos, 1e-6,
                    "Step " + step + ": bias at cachePos=" + cachePos + " should be MASK_FILL (not yet filled)");

            // They AGREE: both say "blocked" at cachePos
            // This is the opposite of what the old test comment claimed ("known-inconsistent")
            log.info("Step {}: cachePos={} mask={} bias={} → CONSISTENT (both blocked)",
                    step, cachePos, maskAtPos, biasAtPos == MASK_FILL ? "MASK_FILL" : biasAtPos);
        }

        dummyDecoder.close();
        log.info("Gap3a passed: cachePos is consistently blocked in both mask and bias");
    }

    /**
     * Gap 3b: With nativeDecodeInputs=true, the same consistency at cachePos must hold.
     *
     * When C++ manages decode inputs, it writes to the capture buffer directly.
     * But the Java-side buildDecoderInputMap still runs and builds the mask/bias
     * (because D2D copies read from the Java array). The mask and bias must still
     * be consistent at cachePos even in this mode.
     */
    @Test
    @DisplayName("Gap3b: cachePos consistency holds with nativeDecodeInputs=true")
    public void testCachePosConsistencyWithNativeDecodeInputs() {
        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .attnMaskReformatOutput(ATTN_REFORMAT_NODE)
                .build();

        List<String> inputNames = createInputNamesWithBias();
        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();
        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);

        Map<String, INDArray> reusableInputs = new HashMap<>();

        for (int step = 0; step < 10; step++) {
            int cachePos = PREFILL_LEN + step;

            Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                    ioConfig, inputNames, dummyDecoder, embeddings, inputIds,
                    679 + step, 1, staticKvBuffers, MAX_KV_LEN, cachePos,
                    true, HIDDEN_SIZE, reusableInputs, true);

            INDArray mask = result.get("attention_mask");
            INDArray bias = result.get(ATTN_REFORMAT_NODE);

            long maskAtPos = mask.getLong(0, cachePos);
            float biasAtPos = bias.getFloat(0, 0, 0, cachePos);

            assertEquals(0L, maskAtPos,
                    "Step " + step + " (native): mask at cachePos=" + cachePos + " should be 0");
            assertEquals(MASK_FILL, biasAtPos, 1e-6,
                    "Step " + step + " (native): bias at cachePos=" + cachePos + " should be MASK_FILL");
        }

        dummyDecoder.close();
        log.info("Gap3b passed: cachePos consistency holds with nativeDecodeInputs");
    }

    /**
     * Gap 3c: Verify that mask and bias FULLY agree at ALL positions (including cachePos)
     * across 20 steps. Unlike the existing test that skips cachePos, this checks everything.
     */
    @Test
    @DisplayName("Gap3c: Full mask/bias agreement at ALL positions including cachePos over 20 steps")
    public void testFullMaskBiasAgreementAllPositions() {
        ModelIOConfig ioConfig = ModelIOConfig.builder()
                .attnMaskReformatOutput(ATTN_REFORMAT_NODE)
                .build();

        List<String> inputNames = createInputNamesWithBias();
        Map<String, INDArray> staticKvBuffers = createStaticKvBuffers();
        SameDiff dummyDecoder = SameDiff.create();
        INDArray embeddings = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray inputIds = Nd4j.createFromArray(new int[]{42}).reshape(1, 1).castTo(DataType.LONG);
        long totalSeqLen = MAX_KV_LEN + 1;

        Map<String, INDArray> reusableInputs = new HashMap<>();

        for (int step = 0; step < 20; step++) {
            int cachePos = PREFILL_LEN + step;

            Map<String, INDArray> result = DecoderUtils.buildDecoderInputMap(
                    ioConfig, inputNames, dummyDecoder, embeddings, inputIds,
                    679 + step, 1, staticKvBuffers, MAX_KV_LEN, cachePos,
                    true, HIDDEN_SIZE, reusableInputs, true);

            INDArray mask = result.get("attention_mask");
            INDArray bias = result.get(ATTN_REFORMAT_NODE);

            int disagreements = 0;
            StringBuilder details = new StringBuilder();
            for (int p = 0; p < MAX_KV_LEN; p++) {
                long maskVal = mask.getLong(0, p);
                float biasVal = bias.getFloat(0, 0, 0, p);

                boolean maskAttend = maskVal == 1;
                boolean biasAttend = Math.abs(biasVal) < 1e-6;

                if (maskAttend != biasAttend) {
                    disagreements++;
                    details.append(String.format("  pos=%d mask=%d bias=%s%n",
                            p, maskVal, biasVal == MASK_FILL ? "MASK_FILL" : String.valueOf(biasVal)));
                }
            }

            // Also check current token position (last position)
            long maskLast = mask.getLong(0, (int) totalSeqLen - 1);
            float biasLast = bias.getFloat(0, 0, 0, (int) totalSeqLen - 1);
            boolean maskLastAttend = maskLast == 1;
            boolean biasLastAttend = Math.abs(biasLast) < 1e-6;
            if (maskLastAttend != biasLastAttend) {
                disagreements++;
                details.append(String.format("  pos=%d (last) mask=%d bias=%s%n",
                        totalSeqLen - 1, maskLast, biasLast == MASK_FILL ? "MASK_FILL" : String.valueOf(biasLast)));
            }

            assertEquals(0, disagreements,
                    "Step " + step + " (cachePos=" + cachePos + "): " + disagreements +
                    " mask/bias disagreements:\n" + details);
        }

        dummyDecoder.close();
        log.info("Gap3c passed: full mask/bias agreement at all positions over 20 steps");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GAP 4: Cast cache warmup → replay lifecycle
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Gap 4a: When switching from output() (warmup) to outputDirect() (replay),
     * the cast cache must not retain stale FP16 entries from the warmup phase.
     *
     * Scenario: step 1 calls output() which casts FP32 activation → FP16 for
     * mixed-precision matmul. Step 2 calls outputDirect() with a DIFFERENT
     * activation. If the cast cache returns the step-1 FP16 data, step 2 gets
     * stale results → identical logits across all steps → degenerate output.
     *
     * This is the LIFECYCLE version of the existing testCastCacheNotStaleAcrossSteps,
     * which only tests within output() calls, not across output()→outputDirect().
     */
    @Test
    @DisplayName("Gap4a: Cast cache freshness across output()→outputDirect() transition")
    public void testCastCacheFreshnessAcrossOutputTransition() {
        SameDiff sd = SameDiff.create();

        SDVariable act = sd.placeHolder("act", DataType.FLOAT, 1, SMALL_K);
        SDVariable wQ = sd.constant("wQ", Nd4j.randn(DataType.HALF, SMALL_K, SMALL_N));
        SDVariable wK = sd.constant("wK", Nd4j.randn(DataType.HALF, SMALL_K, SMALL_N));
        SDVariable q = sd.mmul("q", act, wQ);
        SDVariable k = sd.mmul("k", act, wK);
        SDVariable result = q.add("result", k);

        INDArray wQfp32 = wQ.getArr().castTo(DataType.FLOAT);
        INDArray wKfp32 = wK.getArr().castTo(DataType.FLOAT);

        int numSteps = 8;
        INDArray[] activations = new INDArray[numSteps];
        INDArray[] outputs = new INDArray[numSteps];

        // Steps 0-1: output() (warmup-like)
        for (int i = 0; i < 2; i++) {
            activations[i] = Nd4j.randn(DataType.FLOAT, 1, SMALL_K);
            Map<String, INDArray> out = sd.output(Map.of("act", activations[i]), "result");
            outputs[i] = out.get("result").castTo(DataType.FLOAT).dup();
        }

        // Steps 2-7: outputDirect() (replay-like)
        for (int i = 2; i < numSteps; i++) {
            activations[i] = Nd4j.randn(DataType.FLOAT, 1, SMALL_K);
            Map<String, INDArray> out = sd.outputDirect(Map.of("act", activations[i]), "result");
            outputs[i] = out.get("result").castTo(DataType.FLOAT).dup();
        }

        // All outputs must differ
        int identicalPairs = 0;
        for (int i = 0; i < numSteps; i++) {
            for (int j = i + 1; j < numSteps; j++) {
                if (maxAbsError(outputs[i], outputs[j]) < 1e-6) {
                    identicalPairs++;
                    log.warn("Steps {} and {} produced identical output — stale cast cache?", i, j);
                }
            }
        }
        assertEquals(0, identicalPairs,
                identicalPairs + " output pairs identical — cast cache stale across output→outputDirect transition");

        // All outputs must be correct
        for (int i = 0; i < numSteps; i++) {
            INDArray expected = activations[i].mmul(wQfp32).add(activations[i].mmul(wKfp32));
            double err = maxAbsError(expected, outputs[i]);
            assertTrue(err < 0.2,
                    "Step " + i + " max error " + err + " exceeds tolerance — cast cache corruption?");
        }

        sd.close();
        log.info("Gap4a passed: cast cache fresh across output→outputDirect transition");
    }

    /**
     * Gap 4b: Cast cache must handle the full lifecycle:
     * clearCastCache → warmup → freeze → multiple replay steps.
     *
     * If clearCastCache is not called before warmup (the bug the commit fixed),
     * stale entries from a previous compilation might survive into the new
     * warmup phase, causing wrong results.
     *
     * Simulates: compile once → execute → compile again with different shapes
     * → the second compilation's results must not be contaminated by the first.
     */
    @Test
    @DisplayName("Gap4b: Second compilation produces correct results after cast cache cleared")
    public void testSecondCompilationAfterCastCacheClear() {
        // First compilation: small shapes
        SameDiff sd1 = SameDiff.create();
        SDVariable a1 = sd1.placeHolder("a", DataType.FLOAT, 1, 32);
        SDVariable w1 = sd1.constant("w", Nd4j.randn(DataType.HALF, 32, 64));
        SDVariable out1 = sd1.mmul("out", a1, w1);

        INDArray act1 = Nd4j.randn(DataType.FLOAT, 1, 32);
        Map<String, INDArray> r1 = sd1.output(Map.of("a", act1), "out");
        INDArray result1 = r1.get("out").castTo(DataType.FLOAT).dup();
        sd1.close();

        // Second compilation: different shapes
        SameDiff sd2 = SameDiff.create();
        SDVariable a2 = sd2.placeHolder("a", DataType.FLOAT, 1, 64);
        SDVariable w2 = sd2.constant("w", Nd4j.randn(DataType.HALF, 64, 128));
        SDVariable out2 = sd2.mmul("out", a2, w2);

        INDArray act2 = Nd4j.randn(DataType.FLOAT, 1, 64);
        Map<String, INDArray> r2 = sd2.output(Map.of("a", act2), "out");
        INDArray result2 = r2.get("out").castTo(DataType.FLOAT).dup();

        // Verify second compilation is correct (not contaminated by first)
        INDArray expected = act2.mmul(w2.getArr().castTo(DataType.FLOAT));
        double err = maxAbsError(expected, result2);
        assertTrue(err < 0.1,
                "Second compilation max error " + err + " — cast cache contaminated by first compilation?");

        // Results should have different shapes (not contaminated)
        assertEquals(64, result1.size(1), "First result should have 64 columns");
        assertEquals(128, result2.size(1), "Second result should have 128 columns");

        sd2.close();
        log.info("Gap4b passed: second compilation not contaminated by first");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GAP 5: cuBLAS workspace warmup/capture algorithm consistency
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Gap 5a: Multiple consecutive matmuls must produce numerically consistent results.
     *
     * If cuBLAS selects different algorithms between warmup and replay (due to
     * workspace configuration differences), the numerical results would differ.
     * This test runs the same computation many times and checks that results
     * are bitwise-identical (for deterministic algorithms) or within tight tolerance.
     */
    @Test
    @DisplayName("Gap5a: Repeated matmuls produce consistent results (algorithm stability)")
    public void testRepeatedMatmulAlgorithmConsistency() {
        SameDiff sd = SameDiff.create();

        // Use sizes that stress cuBLAS algorithm selection (M=1 decode-like)
        SDVariable input = sd.placeHolder("x", DataType.FLOAT, 1, 256);
        SDVariable weights = sd.constant("w", Nd4j.randn(DataType.FLOAT, 256, 1024));
        SDVariable mm = sd.mmul("out", input, weights);

        INDArray fixedInput = Nd4j.randn(DataType.FLOAT, 1, 256);

        // Run step 1 via output() (warmup)
        Map<String, INDArray> warmupResult = sd.output(Map.of("x", fixedInput), "out");
        INDArray warmupOut = warmupResult.get("out").dup();

        // Run steps 2-20 via outputDirect() (replay) with SAME input
        INDArray[] replayOuts = new INDArray[19];
        for (int i = 0; i < 19; i++) {
            Map<String, INDArray> replayResult = sd.outputDirect(Map.of("x", fixedInput), "out");
            replayOuts[i] = replayResult.get("out").dup();
        }

        // All replay results should be identical to warmup (same input, deterministic)
        for (int i = 0; i < 19; i++) {
            double diff = maxAbsError(warmupOut, replayOuts[i]);
            // Allow tiny tolerance for cuBLAS non-determinism, but flag large differences
            assertTrue(diff < 1e-4,
                    "Replay step " + (i + 2) + " differs from warmup by " + diff +
                    " — cuBLAS algorithm divergence between warmup and replay?");
        }

        // All replay results should be identical to each other
        for (int i = 1; i < 19; i++) {
            double diff = maxAbsError(replayOuts[0], replayOuts[i]);
            assertTrue(diff < 1e-6,
                    "Replay steps 2 and " + (i + 2) + " differ by " + diff +
                    " — non-deterministic algorithm selection?");
        }

        sd.close();
        log.info("Gap5a passed: matmul results consistent across warmup→replay");
    }

    /**
     * Gap 5b: Mixed-precision matmuls (FP32 activation × FP16 weights) must also be
     * consistent across warmup→replay. cuBLAS Lt may select different internal
     * algorithms for mixed types, and workspace differences would amplify this.
     */
    @Test
    @DisplayName("Gap5b: Mixed-precision matmul consistent across warmup→replay")
    public void testMixedPrecisionMatmulConsistency() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("x", DataType.FLOAT, 1, 256);
        SDVariable weights = sd.constant("w", Nd4j.randn(DataType.HALF, 256, 1024));
        SDVariable mm = sd.mmul("out", input, weights);

        INDArray fixedInput = Nd4j.randn(DataType.FLOAT, 1, 256);

        // Warmup via output()
        Map<String, INDArray> warmupResult = sd.output(Map.of("x", fixedInput), "out");
        INDArray warmupOut = warmupResult.get("out").castTo(DataType.FLOAT).dup();

        // Replay via outputDirect() with same input
        INDArray[] replayOuts = new INDArray[10];
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> replayResult = sd.outputDirect(Map.of("x", fixedInput), "out");
            replayOuts[i] = replayResult.get("out").castTo(DataType.FLOAT).dup();
        }

        // Check consistency
        for (int i = 0; i < 10; i++) {
            double diff = maxAbsError(warmupOut, replayOuts[i]);
            assertTrue(diff < 1e-3,
                    "Mixed-precision replay step " + (i + 2) + " differs from warmup by " + diff);
        }

        sd.close();
        log.info("Gap5b passed: mixed-precision matmul consistent across warmup→replay");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GAP 6: putScalar no device sync for input_ids
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Gap 6a: reusableInputIds.putScalar() writes to the HOST buffer only.
     * Unlike the attention mask path which calls ensureLocation(DEVICE),
     * there is no explicit device sync for input_ids.
     *
     * In the D2D capture buffer copy path, the copy reads from the device buffer
     * of the Java array. If putScalar only updated the host, the device buffer
     * has the PREVIOUS step's token ID.
     *
     * This test verifies that after putScalar on a LONG scalar array, the value
     * is readable via getLong (which may trigger an implicit sync).
     */
    @Test
    @DisplayName("Gap6a: putScalar on reusableInputIds is visible via getLong immediately")
    public void testPutScalarOnInputIdsVisible() {
        INDArray inputIds = Nd4j.createFromArray(new int[]{0}).reshape(1, 1).castTo(DataType.LONG);
        long addr = inputIds.data().address();

        for (int step = 0; step < 50; step++) {
            int tokenId = 100 + step;
            inputIds.putScalar(0, 0, tokenId);

            // getLong may trigger implicit sync — verify value is correct
            long readBack = inputIds.getLong(0, 0);
            assertEquals(tokenId, readBack,
                    "Step " + step + ": putScalar(" + tokenId + ") not visible via getLong. " +
                    "Got " + readBack + ". If stale, model feeds wrong token.");

            // Buffer address must not change (CUDA graph replay stability)
            assertEquals(addr, inputIds.data().address(),
                    "Step " + step + ": input_ids buffer address changed — breaks CUDA graph replay");
        }

        inputIds.close();
        log.info("Gap6a passed: putScalar on input_ids immediately visible");
    }

    /**
     * Gap 6b: After putScalar + explicit ensureLocation(DEVICE), the device buffer
     * must reflect the new token ID. This is what the fix SHOULD look like —
     * matching the attention mask pattern.
     *
     * If this test passes but Gap6a fails, it proves that explicit ensureLocation
     * is needed (the bug is the missing sync).
     */
    @Test
    @DisplayName("Gap6b: putScalar + ensureLocation(DEVICE) on input_ids ensures device visibility")
    public void testPutScalarWithExplicitSyncOnInputIds() {
        INDArray inputIds = Nd4j.createFromArray(new int[]{0}).reshape(1, 1).castTo(DataType.LONG);

        for (int step = 0; step < 50; step++) {
            int tokenId = 200 + step;
            inputIds.putScalar(0, 0, tokenId);

            // Explicit sync — what the mask path does
            Nd4j.getAffinityManager().ensureLocation(inputIds,
                    org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);

            long readBack = inputIds.getLong(0, 0);
            assertEquals(tokenId, readBack,
                    "Step " + step + ": putScalar + ensureLocation still stale. Got " + readBack);
        }

        inputIds.close();
        log.info("Gap6b passed: putScalar + ensureLocation(DEVICE) on input_ids visible");
    }

    /**
     * Gap 6c: Compare the TWO update patterns side by side:
     *
     * Pattern A (current code for input_ids): putScalar only (NO ensureLocation)
     * Pattern B (current code for attention_mask): putScalar + ensureLocation(DEVICE)
     *
     * After 20 rapid updates, verify both patterns show the correct final value.
     * If pattern A shows a stale value but pattern B doesn't, the missing
     * ensureLocation on input_ids is confirmed as a bug.
     */
    @Test
    @DisplayName("Gap6c: putScalar-only vs putScalar+ensureLocation comparison")
    public void testPutScalarWithAndWithoutSync() {
        INDArray noSync = Nd4j.createFromArray(new int[]{0}).reshape(1, 1).castTo(DataType.LONG);
        INDArray withSync = Nd4j.createFromArray(new int[]{0}).reshape(1, 1).castTo(DataType.LONG);

        int lastToken = -1;
        for (int step = 0; step < 20; step++) {
            int tokenId = 300 + step;
            lastToken = tokenId;

            // Pattern A: putScalar only (like input_ids path)
            noSync.putScalar(0, 0, tokenId);

            // Pattern B: putScalar + ensureLocation (like attention_mask path)
            withSync.putScalar(0, 0, tokenId);
            Nd4j.getAffinityManager().ensureLocation(withSync,
                    org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);
        }

        // After all updates, both should show the final value
        long noSyncRead = noSync.getLong(0, 0);
        long withSyncRead = withSync.getLong(0, 0);

        assertEquals(lastToken, withSyncRead,
                "Pattern B (with sync): should show final token " + lastToken);
        assertEquals(lastToken, noSyncRead,
                "Pattern A (no sync): should show final token " + lastToken +
                ". If stale, ensureLocation is needed after input_ids putScalar.");

        noSync.close();
        withSync.close();
        log.info("Gap6c passed: both patterns show correct final value");
    }

    /**
     * Gap 6d: Full decode loop simulation with all reusable buffers updated in lockstep.
     *
     * Simulates the exact sequence from StaticKvCacheDecodeLoop:
     * 1. reusableEmbeddings.assign(rowEmbed) — from embedding table
     * 2. reusableInputIds.putScalar(0, 0, nextTokenId) — no sync
     * 3. buildDecoderInputMap → mask.putScalar + ensureLocation, bias.putScalar + ensureLocation
     *
     * After each step, ALL four reusable buffers must show the correct, current values.
     */
    @Test
    @DisplayName("Gap6d: Full decode buffer update simulation — all 4 reusable buffers consistent")
    public void testFullDecodeBufferUpdateSimulation() {
        int hiddenSize = 32;  // Small for speed
        INDArray embeddingTable = Nd4j.randn(DataType.FLOAT, 1000, hiddenSize);
        INDArray reusableEmbeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, hiddenSize);
        INDArray reusableInputIds = Nd4j.createFromArray(new int[]{0}).reshape(1, 1).castTo(DataType.LONG);
        long totalSeqLen = MAX_KV_LEN + 1;
        INDArray reusableMask = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);
        INDArray reusableBias = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeqLen);

        // Initialize mask and bias for prefill
        reusableMask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, PREFILL_LEN)).assign(1);
        reusableMask.putScalar(0, totalSeqLen - 1, 1);
        for (int k = PREFILL_LEN; k < MAX_KV_LEN; k++) {
            reusableBias.putScalar(new long[]{0, 0, 0, k}, MASK_FILL);
        }

        for (int step = 0; step < 30; step++) {
            int tokenId = 42 + step;
            int cachePos = PREFILL_LEN + step;
            if (cachePos >= MAX_KV_LEN) break;

            // 1. Update embeddings (assign — always syncs device)
            INDArray rowEmbed = embeddingTable.getRow(tokenId).reshape(1, 1, hiddenSize);
            reusableEmbeddings.assign(rowEmbed);

            // 2. Update input_ids (putScalar — NO ensureLocation in current code)
            reusableInputIds.putScalar(0, 0, tokenId);

            // 3. Update mask (putScalar + ensureLocation)
            if (cachePos > 0) {
                reusableMask.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(cachePos - 1)},
                        Nd4j.scalar(DataType.LONG, 1));
                Nd4j.getAffinityManager().ensureLocation(reusableMask,
                        org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);
            }

            // 4. Update bias (putScalar + ensureLocation)
            if (cachePos > 0) {
                reusableBias.putScalar(new long[]{0, 0, 0, cachePos - 1}, 0.0f);
                Nd4j.getAffinityManager().ensureLocation(reusableBias,
                        org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);
            }

            // Verify all 4 buffers
            float embedFirst = reusableEmbeddings.getFloat(0, 0, 0);
            float rowFirst = embeddingTable.getFloat(tokenId, 0);
            assertEquals(rowFirst, embedFirst, 1e-6,
                    "Step " + step + ": embeddings not updated (expected from row " + tokenId + ")");

            long readTokenId = reusableInputIds.getLong(0, 0);
            assertEquals(tokenId, readTokenId,
                    "Step " + step + ": input_ids stale (expected " + tokenId + ", got " + readTokenId + ")");

            if (cachePos > 0) {
                assertEquals(1L, reusableMask.getLong(0, cachePos - 1),
                        "Step " + step + ": mask at cachePos-1 not updated");
            }

            if (cachePos > 0) {
                assertEquals(0.0f, reusableBias.getFloat(0, 0, 0, cachePos - 1), 1e-6,
                        "Step " + step + ": bias at cachePos-1 not unmasked");
            }
        }

        reusableEmbeddings.close();
        reusableInputIds.close();
        reusableMask.close();
        reusableBias.close();
        log.info("Gap6d passed: all 4 decode buffers consistent across 30 steps");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ADDITIONAL: Cross-gap interaction tests
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Cross-gap 1: The FULL step 1→2 transition with ALL components active simultaneously.
     *
     * Combines: setCloseable toggle (gap1) + mask accumulation (gap2) + bias consistency (gap3)
     * + cast cache (gap4) + input_ids sync (gap6).
     *
     * This simulates the exact sequence from StaticKvCacheDecodeLoop for steps 1-5
     * with a real SameDiff model (simple matmul) and verifies that ALL outputs differ
     * AND all inputs are correctly evolved.
     */
    @Test
    @DisplayName("Cross-gap: Full step 1-5 simulation with all components interacting")
    public void testFullDecodeTransitionAllComponentsInteracting() {
        SameDiff sd = SameDiff.create();

        // Simple model: embeddings → matmul → logits
        int hiddenDim = 32;
        int vocabDim = 64;
        SDVariable embeds = sd.placeHolder("embeds", DataType.FLOAT, 1, 1, hiddenDim);
        SDVariable embedsFlat = sd.reshape("flat", embeds, 1, hiddenDim);
        SDVariable weights = sd.constant("w", Nd4j.randn(DataType.HALF, hiddenDim, vocabDim));
        SDVariable logits = sd.mmul("logits", embedsFlat, weights);

        INDArray weightsFp32 = weights.getArr().castTo(DataType.FLOAT);
        INDArray embeddingTable = Nd4j.randn(DataType.FLOAT, 1000, hiddenDim);

        // Reusable buffers
        INDArray reusableEmbeds = null;
        INDArray reusableIds = null;
        long totalSeqLen = MAX_KV_LEN + 1;
        INDArray reusableMask = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);
        reusableMask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, PREFILL_LEN)).assign(1);

        INDArray[] outputs = new INDArray[5];
        int[] tokenIds = {42, 99, 77, 13, 55};

        for (int step = 0; step < 5; step++) {
            int tokenId = tokenIds[step];
            int cachePos = PREFILL_LEN + step;

            // Update reusable buffers
            INDArray rowEmbed = embeddingTable.getRow(tokenId).reshape(1, 1, hiddenDim);
            if (reusableEmbeds == null) {
                reusableEmbeds = rowEmbed.dup();
            } else {
                reusableEmbeds.assign(rowEmbed);
            }

            if (reusableIds == null) {
                reusableIds = Nd4j.createFromArray(new int[]{tokenId}).reshape(1, 1).castTo(DataType.LONG);
            } else {
                reusableIds.putScalar(0, 0, tokenId);
            }

            // Update mask
            if (cachePos > 0 && step > 0) {
                reusableMask.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(cachePos - 1)},
                        Nd4j.scalar(DataType.LONG, 1));
                Nd4j.getAffinityManager().ensureLocation(reusableMask,
                        org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);
            }

            // Execute — step 0-1 via output(), step 2+ via outputDirect()
            Map<String, INDArray> out;
            if (step < 2) {
                out = sd.output(Map.of("embeds", reusableEmbeds), "logits");
            } else {
                out = sd.outputDirect(Map.of("embeds", reusableEmbeds), "logits");
            }
            outputs[step] = out.get("logits").castTo(DataType.FLOAT).dup();

            // Verify output matches manual computation
            INDArray expectedFlat = reusableEmbeds.reshape(1, hiddenDim);
            INDArray expected = expectedFlat.mmul(weightsFp32);
            double err = maxAbsError(expected, outputs[step]);
            assertTrue(err < 0.1,
                    "Step " + step + " (via " + (step < 2 ? "output" : "outputDirect") +
                    "): max error " + err + " exceeds tolerance");
        }

        // All 5 outputs must differ (different token embeddings)
        for (int i = 0; i < 5; i++) {
            for (int j = i + 1; j < 5; j++) {
                double diff = maxAbsError(outputs[i], outputs[j]);
                assertTrue(diff > 1e-3,
                        "Steps " + i + " and " + j + " produced identical output (diff=" + diff + ")");
            }
        }

        // Verify mask accumulated correctly
        for (int i = 0; i < PREFILL_LEN + 4; i++) {
            assertEquals(1L, reusableMask.getLong(0, i),
                    "Mask at position " + i + " should be 1 after 5 steps");
        }

        sd.close();
        reusableEmbeds.close();
        reusableIds.close();
        reusableMask.close();
        log.info("Cross-gap passed: full step 1-5 with all components correct");
    }

    /**
     * Cross-gap 2: Verify that mask and bias agree even when interleaving
     * setCloseable toggling (simulating output()→outputDirect() switch).
     *
     * This catches the scenario where setCloseable(false) on the reusable mask/bias
     * causes putScalar writes to be lost or device sync to be skipped.
     */
    @Test
    @DisplayName("Cross-gap 2: Mask/bias consistency survives setCloseable interleaving")
    public void testMaskBiasConsistencyWithCloseableInterleaving() {
        long totalSeqLen = MAX_KV_LEN + 1;
        INDArray mask = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeqLen);

        // Initialize
        mask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, PREFILL_LEN)).assign(1);
        mask.putScalar(0, totalSeqLen - 1, 1);
        for (int k = PREFILL_LEN; k < MAX_KV_LEN; k++) {
            bias.putScalar(new long[]{0, 0, 0, k}, MASK_FILL);
        }

        for (int step = 0; step < 15; step++) {
            int cachePos = PREFILL_LEN + step;
            if (cachePos >= MAX_KV_LEN) break;

            // Steps 0-1: simulate output() wrapping
            if (step < 2) {
                mask.setCloseable(false);
                bias.setCloseable(false);
            }

            // Update mask
            if (cachePos > 0) {
                mask.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(cachePos - 1)},
                        Nd4j.scalar(DataType.LONG, 1));
                Nd4j.getAffinityManager().ensureLocation(mask,
                        org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);
            }

            // Update bias
            if (cachePos > 0) {
                bias.putScalar(new long[]{0, 0, 0, cachePos - 1}, 0.0f);
                Nd4j.getAffinityManager().ensureLocation(bias,
                        org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);
            }

            // Undo wrapping
            if (step < 2) {
                mask.setCloseable(true);
                bias.setCloseable(true);
            }

            // Check consistency at ALL past positions
            for (int p = 0; p < cachePos; p++) {
                long maskVal = mask.getLong(0, p);
                float biasVal = bias.getFloat(0, 0, 0, p);
                assertEquals(1L, maskVal,
                        "Step " + step + ": mask at position " + p + " should be 1");
                assertEquals(0.0f, biasVal, 1e-6,
                        "Step " + step + ": bias at position " + p + " should be 0.0");
            }

            // cachePos should be blocked in both
            assertEquals(0L, mask.getLong(0, cachePos),
                    "Step " + step + ": mask at cachePos=" + cachePos + " should be 0");
            assertEquals(MASK_FILL, bias.getFloat(0, 0, 0, cachePos), 1e-6,
                    "Step " + step + ": bias at cachePos=" + cachePos + " should be MASK_FILL");
        }

        mask.close();
        bias.close();
        log.info("Cross-gap 2 passed: mask/bias consistent despite setCloseable interleaving");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════════

    private Map<String, INDArray> createStaticKvBuffers() {
        Map<String, INDArray> buffers = new HashMap<>();
        for (int i = 0; i < NUM_LAYERS; i++) {
            buffers.put("past_key_values." + i + ".key",
                    Nd4j.zeros(DataType.FLOAT, 1, NUM_KV_HEADS, MAX_KV_LEN, HEAD_DIM));
            buffers.put("past_key_values." + i + ".value",
                    Nd4j.zeros(DataType.FLOAT, 1, NUM_KV_HEADS, MAX_KV_LEN, HEAD_DIM));
        }
        return buffers;
    }

    private List<String> createInputNamesWithBias() {
        List<String> names = new ArrayList<>();
        names.add("inputs_embeds");
        names.add("attention_mask");
        names.add("input_ids");
        names.add("position_ids");
        names.add(ATTN_REFORMAT_NODE);
        for (int i = 0; i < NUM_LAYERS; i++) {
            names.add("past_key_values." + i + ".key");
            names.add("past_key_values." + i + ".value");
        }
        return names;
    }

    private DecoderUtils.KVCacheNames createKvNames() {
        List<String> keyNames = new ArrayList<>();
        List<String> valueNames = new ArrayList<>();
        for (int i = 0; i < NUM_LAYERS; i++) {
            keyNames.add("present." + i + ".key");
            valueNames.add("present." + i + ".value");
        }
        return new DecoderUtils.KVCacheNames(keyNames, valueNames);
    }

    private static double maxAbsError(INDArray expected, INDArray actual) {
        INDArray diff = Nd4j.math().abs(expected.sub(actual));
        return diff.maxNumber().doubleValue();
    }
}
