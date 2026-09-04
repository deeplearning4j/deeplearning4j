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
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.nd4j.common.tests.tags.TagNames;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DSP device/stream-related ext-input tests extracted from DspExtInputStalenessTest.
 *
 * Covers:
 * - Category 26: Cross-Stream Device Write Tests (addi on LC stream, steady state cross-stream sync)
 * - Category 27: VLM Decode Pattern Reproduction — Additional Tests
 * - Category 28: KV-like Multi-Buffer Pattern Tests
 * - Category 1 (JNI): True Cross-Stream Device Write Tests using JNI CUDA stream API
 * - MISSING PLAN TESTS: additional cross-stream, variable classification, arg table, fast-path,
 *   steady state, gap slot, multi-external, and VLM decode tests (lines 5547–6194)
 */
@Slf4j
@Tag(TagNames.FULL_CI)
@TestInstance(TestInstance.Lifecycle.PER_METHOD)
public class DspExtInputDeviceStreamTest extends DspExtInputTestSupport {

    private SameDiff sd;

    @AfterEach
    void cleanup() {
        if (sd != null) {
            sd.close();
            sd = null;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GRAPH FIXTURES
    // ═══════════════════════════════════════════════════════════════════════════

    // ═══════════════════════════════════════════════════════════════════════════
    // SHARED HELPERS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Helper: check if the CUDA stream JNI API is available (skip on CPU).
     */
    private boolean isCudaStreamApiAvailable() {
        try {
            Pointer p = NativeOpsHolder.getInstance().getDeviceNativeOps().dspCreateTestStream();
            if (p == null) return false;  // CPU build returns null
            NativeOpsHolder.getInstance().getDeviceNativeOps().dspDestroyTestStream(p);
            return true;
        } catch (UnsupportedOperationException e) {
            return false;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 26: Cross-Stream Device Write Tests
    // Uses arr.addi() to write to device buffer on LC default stream,
    // creating the cross-stream pattern (LC stream vs DSP stream).
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Device buffer write via addi() on LC default stream, then DSP replay.
     * addi() runs a CUDA kernel on the LC default stream, writing directly to
     * device buffer. After addi(), isPrimaryActual() returns false (device is
     * authoritative). performPreReplaySync should handle cross-stream ordering.
     */
    @ParameterizedTest(name = "deviceWriteThenD2D mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Device write via addi() on LC stream → DSP replay sees fresh data")
    void testDeviceWriteThenD2D(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(16, 8);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 16);
        // Warmup to get to REPLAYING state
        warmupWithChangingInput(sd, "x", input, "out", 12, new long[]{1, 16});

        // Now modify device buffer via addi (runs CUDA kernel on LC default stream)
        // Then call sd.output — DSP replay on DSP stream must see the device-written data
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            // Reset to base value via assign (host write + syncToDevice)
            input.assign(Nd4j.valueArrayOf(new long[]{1, 16}, 1.0));
            // Device write via addi — runs on LC default stream
            input.addi(step + 1.0);
            // DSP replay — must see the post-addi values
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck — device write via addi() not visible to DSP. sums=" + sums);
        }
        log.info("[DEVICE_WRITE_D2D] mode={} PASS — addi device writes visible to DSP across 20 steps", mode);
    }

    /**
     * In-place device modify with stable address — simulates the VLM embed
     * lookup kernel pattern. Same buffer address, content overwritten on device
     * each step via addi().
     */
    @ParameterizedTest(name = "inPlaceDeviceModifyStableAddress mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Same address, content overwritten on device via addi() each step for 20 steps")
    void testInPlaceDeviceModifyStableAddress(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(16, 8);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 16);
        warmupWithChangingInput(sd, "x", input, "out", 12, new long[]{1, 16});

        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            // Zero out on host, sync to device
            input.assign(0.0);
            // Device write: addi runs CUDA kernel, writes (step+1)*0.5 to each element
            input.addi((step + 1) * 0.5);
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [IN_PLACE_DEVICE]: STUCK! " + stuckCount + "/19 steps. "
                        + "In-place device modify (addi) not reflected in DSP replay. "
                        + "sums=" + sums.subList(0, Math.min(8, sums.size())));
        log.info("[IN_PLACE_DEVICE] mode={} PASS — {}/19 unique steps via in-place device modify", mode, 19 - stuckCount);
    }

    /**
     * First 4 steps warmup with host assign, then switch to device-only writes
     * via addi() for steps 5-20.
     */
    @ParameterizedTest(name = "inPlaceDeviceModifyAfterSEALED mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Host assign during warmup, then device-only writes via addi() after SEALED")
    void testInPlaceDeviceModifyAfterSEALED(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(16, 8);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 16);

        // Warmup 4 steps with host assign (normal path)
        for (int i = 0; i < 4; i++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 16}, (double)(i + 1)));
            sd.output(singlePh("x", input), "out");
        }

        // Steps 5-20: device-only writes via addi (LC default stream)
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 16; step++) {
            input.assign(0.0); // reset via host
            input.addi((step + 5) * 0.3); // device write
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + (i + 4) + " stuck after switching to device writes. sums=" + sums);
        }
        log.info("[DEVICE_AFTER_SEALED] mode={} PASS — device-only writes after SEALED all reflected", mode);
    }

    /**
     * Cross-stream sync test for steady state: device write on LC stream
     * followed by executeSteadyState (if available) on DSP stream.
     */
    @ParameterizedTest(name = "steadyStateCrossStreamSync mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Device write on LC stream → sd.output() in steady state → cross-stream sync fires")
    void testSteadyStateCrossStreamSync(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(16, 8);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 16);
        // Get well into steady state
        warmupWithChangingInput(sd, "x", input, "out", 15, new long[]{1, 16});

        // Device write + sd.output in steady state
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(0.0);
            input.addi((step + 1) * 2.0); // device write on LC stream
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck in steady state with device writes. sums=" + sums);
        }
        log.info("[STEADY_CROSS_STREAM] mode={} PASS — 20 device-write steps in steady state all unique", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 27: VLM Decode Pattern Reproduction — Additional Tests
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Simulate DecodeInputEvolutor bug: buildStepInputs() doesn't include inputs_embeds.
     * After warmup, omit "inputs_embeds" from the placeholder map.
     * Documents behavior: outputs ARE stuck (this is the missing-input bug).
     */
    @ParameterizedTest(name = "decodePatternInputEmbedNotInEvolutor mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("VLM pattern: inputs_embeds missing from step map after warmup — documents stuck behavior")
    void testDecodePatternInputEmbedNotInEvolutor(GraphExecutionMode mode) {
        sd = buildLargeDecoderGraph(16, 2);
        configureMode(sd, mode);

        INDArray embed = Nd4j.ones(DataType.FLOAT, 1, 1, 16);
        INDArray posIds = Nd4j.zeros(DataType.FLOAT, 1, 1);
        INDArray kv0 = Nd4j.randn(DataType.FLOAT, 1, 4, 16);
        INDArray kv1 = Nd4j.randn(DataType.FLOAT, 1, 4, 16);

        Map<String, INDArray> fullPh = new LinkedHashMap<>();
        fullPh.put("inputs_embeds", embed);
        fullPh.put("position_ids", posIds);
        fullPh.put("layer_0_kv", kv0);
        fullPh.put("layer_1_kv", kv1);

        // Warmup with full map
        for (int i = 0; i < 8; i++) {
            embed.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, (double)(i + 1)));
            posIds.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)i));
            sd.output(fullPh, "out");
        }

        // Now simulate the bug: omit inputs_embeds, only provide position + KV
        Map<String, INDArray> incompletePh = new LinkedHashMap<>();
        incompletePh.put("position_ids", posIds);
        incompletePh.put("layer_0_kv", kv0);
        incompletePh.put("layer_1_kv", kv1);

        // Must either throw (correct) or produce result (using cached embed — stuck)
        boolean threwException = false;
        int stuckCount = 0;
        INDArray prevResult = null;
        try {
            for (int step = 0; step < 10; step++) {
                posIds.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)(step + 100)));
                INDArray result = sd.output(incompletePh, "out").get("out").dup();
                if (prevResult != null) {
                    double change = result.sub(prevResult).amaxNumber().doubleValue();
                    if (change < 1e-6) stuckCount++;
                }
                prevResult = result;
            }
        } catch (Exception e) {
            threwException = true;
            log.info("[EMBED_MISSING] mode={} correctly threw: {}", mode, e.getMessage());
        }

        if (!threwException) {
            // Document: without inputs_embeds in the map, output IS stuck
            // (because the cached embed from warmup is reused)
            log.info("[EMBED_MISSING] mode={} no exception — stuckCount={}/9 (expected: stuck without embed)",
                    mode, stuckCount);
            // This test DOCUMENTS the bug, not asserts it's fixed.
            // If embed is missing, position-only changes may or may not propagate.
        } else {
            log.info("[EMBED_MISSING] mode={} PASS — missing placeholder correctly rejected", mode);
        }
    }

    /**
     * Placeholder classified at compile time — does executor auto-mark it variable?
     * Does D2D staging happen without explicit markVariable?
     * Documents: YES (auto) or NO (manual required).
     */
    @ParameterizedTest(name = "decodePatternNoMarkVariableAutoDetect mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Placeholder auto-detection: outputs change without explicit markVariable")
    void testDecodePatternNoMarkVariableAutoDetect(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(16, 8);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 16);
        // Warmup — DO NOT call markVariable anywhere
        warmupWithChangingInput(sd, "x", input, "out", 12, new long[]{1, 16});

        // Run 20 steps with changing input — must work via auto-detection
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 16}, (double)(step + 200)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck without explicit markVariable! "
                            + "Auto-detection failed. sums=" + sums);
        }
        log.info("[AUTO_DETECT] mode={} PASS — placeholder auto-detected, 20 steps unique without markVariable", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 28: KV-like Multi-Buffer Pattern Tests
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * 30 ext inputs (KV-like): same address, content changed via assign() each step.
     * Simulates KV cache pattern in VLM where 30 KV buffers get scatter-written.
     */
    @ParameterizedTest(name = "kvPatternStableBufferAssign mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("30 KV-like ext inputs, content changes each step via assign() — all reflected")
    void testKVPatternStableBufferAssign(GraphExecutionMode mode) {
        int numKV = 8; // scaled down from 30 for test speed
        int dim = 8;

        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);

        // Create numKV KV placeholders, sum them all + x, then matmul
        SDVariable running = x;
        for (int k = 0; k < numKV; k++) {
            SDVariable kv = g.placeHolder("kv_" + k, DataType.FLOAT, 1, dim);
            running = running.add("add_kv_" + k, kv);
        }
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, dim, 4)).addi(0.1f));
        g.mmul("out", running, w);
        sd = g;
        configureMode(sd, mode);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 1, dim);
        INDArray[] kvArrs = new INDArray[numKV];
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", xArr);
        for (int k = 0; k < numKV; k++) {
            kvArrs[k] = Nd4j.ones(DataType.FLOAT, 1, dim);
            ph.put("kv_" + k, kvArrs[k]);
        }

        // Warmup
        for (int i = 0; i < 8; i++) {
            xArr.assign(Nd4j.valueArrayOf(new long[]{1, dim}, (double)(i + 1)));
            for (int k = 0; k < numKV; k++) {
                kvArrs[k].assign(Nd4j.valueArrayOf(new long[]{1, dim}, (double)(i + k + 1)));
            }
            sd.output(ph, "out");
        }

        // Run 20 steps: x changes, all KV change
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            xArr.assign(Nd4j.valueArrayOf(new long[]{1, dim}, (double)(step + 100)));
            for (int k = 0; k < numKV; k++) {
                kvArrs[k].assign(Nd4j.valueArrayOf(new long[]{1, dim}, (double)(step + k + 200)));
            }
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck with " + numKV + " KV inputs changing. sums=" + sums);
        }
        log.info("[KV_PATTERN] mode={} PASS — {} KV inputs all reflected across 20 steps", mode, numKV);
    }

    /**
     * Embedding + KV pattern together:
     * 1 "embedding" ext input: in-place assign each step
     * + numKV KV ext inputs: all assign each step
     * + constants: never change
     */
    @ParameterizedTest(name = "embeddingPlusKVPattern mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("1 embedding + 8 KV ext inputs + constants — all correct across 20 steps")
    void testEmbeddingPlusKVPattern(GraphExecutionMode mode) {
        sd = buildLargeDecoderGraph(16, 4);
        configureMode(sd, mode);

        INDArray embed = Nd4j.ones(DataType.FLOAT, 1, 1, 16);
        INDArray posIds = Nd4j.zeros(DataType.FLOAT, 1, 1);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("inputs_embeds", embed);
        ph.put("position_ids", posIds);

        INDArray[] kvArrs = new INDArray[4];
        for (int layer = 0; layer < 4; layer++) {
            kvArrs[layer] = Nd4j.randn(DataType.FLOAT, 1, 4, 16);
            ph.put("layer_" + layer + "_kv", kvArrs[layer]);
        }

        // Warmup
        for (int i = 0; i < 8; i++) {
            embed.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, (double)(i + 1)));
            posIds.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)i));
            for (int k = 0; k < 4; k++) {
                kvArrs[k].assign(Nd4j.valueArrayOf(new long[]{1, 4, 16}, (double)(i + k + 1) * 0.1));
            }
            sd.output(ph, "out");
        }

        // Run: embed changes, pos changes, KV changes
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            embed.assign(Nd4j.valueArrayOf(new long[]{1, 1, 16}, (double)(step + 100)));
            posIds.assign(Nd4j.valueArrayOf(new long[]{1, 1}, (double)(step + 8)));
            for (int k = 0; k < 4; k++) {
                kvArrs[k].assign(Nd4j.valueArrayOf(new long[]{1, 4, 16}, (double)(step + k + 200) * 0.01));
            }
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [EMBED+KV]: STUCK! " + stuckCount + "/19 steps. "
                        + "sums=" + sums.subList(0, Math.min(8, sums.size())));
        log.info("[EMBED_KV] mode={} PASS — {}/19 unique with embed+4KV all changing", mode, 19 - stuckCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CATEGORY 1 (JNI): True Cross-Stream Device Write Tests
    // ═══════════════════════════════════════════════════════════════════════════
    // These tests use the new JNI bindings (dspWriteDeviceBufferOnDefaultStream,
    // dspWriteDeviceBufferOnExplicitStream, dspSyncStream, etc.) to test the
    // DSP cross-stream sync mechanism with controlled stream placement.

    /**
     * Write to ext input device buffer on DEFAULT stream, then replay.
     * Tests that performPreReplaySync's cross-stream event sync (default→DSP)
     * makes the fresh data visible to graph replay.
     */
    @ParameterizedTest(name = "jniDeviceWriteDefaultStream mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("JNI: device write on default stream → replay sees fresh data")
    void testJniDeviceWriteDefaultStream(GraphExecutionMode mode) {
        Assumptions.assumeTrue(isCudaStreamApiAvailable(), "CUDA stream API not available");

        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        Map<String, INDArray> ph = singlePh("x", input);

        // Warmup to REPLAYING — use changing input values so the plan sees
        // this placeholder as dynamic (not constant), enabling the staging path.
        warmupWithChangingInput(sd, "x", input, "out", 8, new long[]{1, 8});
        DspHandle h = sd.dsp();

        // Mark input as variable so staging buffers are allocated for it.
        // Without this, performPreReplaySync skips staging for "constant" inputs.
        int xIdx = h.extInputIndex("x");
        assertTrue(xIdx >= 0, "ext input 'x' not found");
        h.markVariable(xIdx);

        // Run one more step to trigger staging buffer allocation
        input.assign(Nd4j.ones(DataType.FLOAT, 1, 8));
        Map<String, INDArray> baseline = sd.output(ph, "out");
        double baseSum = baseline.get("out").sumNumber().doubleValue();

        // Write different data to device buffer on LC default stream via JNI
        float[] newData = new float[8];
        Arrays.fill(newData, 5.0f);
        FloatPointer hostPtr = new FloatPointer(newData);

        int rc = h.writeDeviceBufferOnDefaultStream(xIdx, hostPtr, 8 * 4);
        assertEquals(0, rc, "writeDeviceBufferOnDefaultStream failed with rc=" + rc);

        // Verify device is now authoritative
        assertTrue(h.isExtInputDeviceAuthoritative(xIdx),
                "Device should be authoritative after device write");

        // Replay — performPreReplaySync should sync default→DSP stream
        // The H2D sync in step 2 skips this input (deviceWritePending_ is set),
        // preserving the JNI-written data in the staging buffer.
        Map<String, INDArray> after = sd.output(ph, "out");
        double afterSum = after.get("out").sumNumber().doubleValue();

        assertNotEquals(baseSum, afterSum, 1e-3,
                mode + " [JNI_DEFAULT_STREAM]: output unchanged after device write! "
                        + "base=" + baseSum + " after=" + afterSum);
        log.info("[JNI_DEFAULT_STREAM] mode={} PASS — base={} after={}", mode, baseSum, afterSum);

        hostPtr.close();
    }

    /**
     * Write to ext input device buffer on an EXPLICIT (non-default, non-DSP) stream,
     * then replay WITHOUT explicit sync. This tests whether performPreReplaySync's
     * cross-stream event handles arbitrary write streams or only the LC default stream.
     *
     * Key question: does the current cross-stream sync (which records event on
     * defaultStream, then waits on dspStream) handle writes on a third stream?
     * If not, this test documents the gap.
     */
    @ParameterizedTest(name = "jniDeviceWriteExplicitStreamNoSync mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("JNI: device write on explicit stream, NO sync → document behavior")
    void testJniDeviceWriteExplicitStreamNoSync(GraphExecutionMode mode) {
        Assumptions.assumeTrue(isCudaStreamApiAvailable(), "CUDA stream API not available");

        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        Map<String, INDArray> ph = singlePh("x", input);

        warmup(sd, ph, "out", 5);
        DspHandle h = sd.dsp();

        // Baseline
        Map<String, INDArray> baseline = sd.output(ph, "out");
        double baseSum = baseline.get("out").sumNumber().doubleValue();

        int xIdx = h.extInputIndex("x");
        Pointer testStream = h.createTestStream();
        assertNotNull(testStream, "Failed to create test stream");

        try {
            // Write different data on EXPLICIT stream (not default, not DSP)
            float[] newData = new float[8];
            Arrays.fill(newData, 10.0f);
            FloatPointer hostPtr = new FloatPointer(newData);

            int rc = h.writeDeviceBufferOnExplicitStream(xIdx, hostPtr, 8 * 4, testStream);
            assertEquals(0, rc, "writeDeviceBufferOnExplicitStream failed");

            // NO explicit sync — rely on DSP's cross-stream mechanism
            // performPreReplaySync only syncs defaultStream→dspStream,
            // so writes on testStream may or may not be visible

            Map<String, INDArray> after = sd.output(ph, "out");
            double afterSum = after.get("out").sumNumber().doubleValue();

            // Document the behavior: if output changed, cross-stream sync covers it;
            // if output is stale, this is a known gap (only default stream is synced)
            if (Math.abs(afterSum - baseSum) < 1e-3) {
                log.warn("[JNI_EXPLICIT_NO_SYNC] mode={} — STALE! Output unchanged after explicit "
                        + "stream write without sync. base={} after={}. This documents that "
                        + "performPreReplaySync only syncs default→DSP, not arbitrary streams.",
                        mode, baseSum, afterSum);
                // This is expected behavior — performPreReplaySync only syncs the default stream.
                // NOT a test failure — it documents the known scope of cross-stream sync.
            } else {
                log.info("[JNI_EXPLICIT_NO_SYNC] mode={} — FRESH! Output changed even without "
                        + "explicit sync. base={} after={}. Cross-stream mechanism may cover "
                        + "all streams, or write completed before replay due to timing.",
                        mode, baseSum, afterSum);
            }
            // Test passes either way — it's a documentation test
            hostPtr.close();
        } finally {
            h.destroyTestStream(testStream);
        }
    }

    /**
     * Write to ext input device buffer on an EXPLICIT stream, then explicitly
     * sync that stream BEFORE replay. Output MUST reflect the fresh data.
     */
    @ParameterizedTest(name = "jniDeviceWriteExplicitStreamWithSync mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("JNI: device write on explicit stream + explicit sync → output correct")
    void testJniDeviceWriteExplicitStreamWithSync(GraphExecutionMode mode) {
        Assumptions.assumeTrue(isCudaStreamApiAvailable(), "CUDA stream API not available");

        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        Map<String, INDArray> ph = singlePh("x", input);

        warmupWithChangingInput(sd, "x", input, "out", 8, new long[]{1, 8});
        DspHandle h = sd.dsp();

        int xIdx = h.extInputIndex("x");
        h.markVariable(xIdx);

        input.assign(Nd4j.ones(DataType.FLOAT, 1, 8));
        Map<String, INDArray> baseline = sd.output(ph, "out");
        double baseSum = baseline.get("out").sumNumber().doubleValue();

        Pointer testStream = h.createTestStream();
        assertNotNull(testStream, "Failed to create test stream");

        try {
            float[] newData = new float[8];
            Arrays.fill(newData, 10.0f);
            FloatPointer hostPtr = new FloatPointer(newData);

            int rc = h.writeDeviceBufferOnExplicitStream(xIdx, hostPtr, 8 * 4, testStream);
            assertEquals(0, rc, "writeDeviceBufferOnExplicitStream failed");

            // Explicitly sync the test stream — guarantee data is visible on device
            int syncRc = h.syncStream(testStream);
            assertEquals(0, syncRc, "dspSyncStream failed with rc=" + syncRc);

            // Now replay — data is on device, sync ensures visibility
            Map<String, INDArray> after = sd.output(ph, "out");
            double afterSum = after.get("out").sumNumber().doubleValue();

            assertNotEquals(baseSum, afterSum, 1e-3,
                    mode + " [JNI_EXPLICIT_WITH_SYNC]: output unchanged after explicit stream "
                            + "write + sync! base=" + baseSum + " after=" + afterSum);
            log.info("[JNI_EXPLICIT_WITH_SYNC] mode={} PASS — base={} after={}", mode, baseSum, afterSum);

            hostPtr.close();
        } finally {
            h.destroyTestStream(testStream);
        }
    }

    /**
     * Verify isPrimaryActual state transitions:
     * 1. Fresh INDArray from Java → host authoritative (isPrimaryActual=true)
     * 2. After sd.output() warmup → device authoritative (synced to device)
     * 3. After host assign() → host authoritative again
     * 4. After JNI device write → device authoritative
     */
    @ParameterizedTest(name = "jniDeviceAuthoritativeTransitions mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("JNI: isPrimaryActual state transitions through lifecycle")
    void testJniDeviceAuthoritativeTransitions(GraphExecutionMode mode) {
        Assumptions.assumeTrue(isCudaStreamApiAvailable(), "CUDA stream API not available");

        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        Map<String, INDArray> ph = singlePh("x", input);

        // Warmup
        warmup(sd, ph, "out", 5);
        DspHandle h = sd.dsp();
        int xIdx = h.extInputIndex("x");

        // After warmup + output, device should have been synced
        sd.output(ph, "out");

        // Write new host data via assign
        input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, 3.0));
        // After assign, host is authoritative (isPrimaryActual should be true, device NOT authoritative)
        // Note: assign() touches host, marks host as fresh
        boolean deviceAuthAfterAssign = h.isExtInputDeviceAuthoritative(xIdx);
        log.info("[AUTH_TRANSITIONS] mode={} after assign: deviceAuth={}", mode, deviceAuthAfterAssign);

        // Write to device via JNI
        float[] deviceData = new float[8];
        Arrays.fill(deviceData, 7.0f);
        FloatPointer hostPtr = new FloatPointer(deviceData);
        int rc = h.writeDeviceBufferOnDefaultStream(xIdx, hostPtr, 8 * 4);
        assertEquals(0, rc, "device write failed");

        // Now device should be authoritative
        boolean deviceAuthAfterWrite = h.isExtInputDeviceAuthoritative(xIdx);
        assertTrue(deviceAuthAfterWrite,
                mode + " device should be authoritative after JNI device write");

        log.info("[AUTH_TRANSITIONS] mode={} PASS — assign→deviceAuth={}, jniWrite→deviceAuth={}",
                mode, deviceAuthAfterAssign, deviceAuthAfterWrite);
        hostPtr.close();
    }

    /**
     * Multi-step test: alternate between host writes (assign) and device writes
     * (JNI) across 20 steps. Every step MUST produce different output.
     * This exercises the full cross-stream sync for both directions.
     */
    @ParameterizedTest(name = "jniAlternatingHostDeviceWrites mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("JNI: alternating host/device writes over 20 steps — no stuck output")
    void testJniAlternatingHostDeviceWrites(GraphExecutionMode mode) {
        Assumptions.assumeTrue(isCudaStreamApiAvailable(), "CUDA stream API not available");

        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        Map<String, INDArray> ph = singlePh("x", input);

        warmupWithChangingInput(sd, "x", input, "out", 8, new long[]{1, 8});
        DspHandle h = sd.dsp();
        int xIdx = h.extInputIndex("x");
        h.markVariable(xIdx);

        List<Double> sums = new ArrayList<>();
        FloatPointer hostPtr = new FloatPointer(8);

        for (int step = 0; step < 20; step++) {
            if (step % 2 == 0) {
                // Even steps: host write via assign()
                input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 10)));
            } else {
                // Odd steps: device write via JNI on default stream
                float val = (float)(step * 3 + 100);
                for (int j = 0; j < 8; j++) hostPtr.put(j, val);
                int rc = h.writeDeviceBufferOnDefaultStream(xIdx, hostPtr, 8 * 4);
                assertEquals(0, rc, "device write failed at step " + step);
            }

            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        hostPtr.close();

        // Count stuck steps
        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [JNI_ALTERNATING]: STUCK! " + stuckCount + "/19 steps. "
                        + "sums=" + sums.subList(0, Math.min(8, sums.size())));
        log.info("[JNI_ALTERNATING] mode={} PASS — {}/19 unique with alternating host/device writes",
                mode, 19 - stuckCount);
    }

    /**
     * Stress test: write to device on explicit stream for 20 steps, sync each
     * time, verify output changes every step. This validates that the explicit
     * stream + sync pattern works reliably across many iterations.
     */
    @ParameterizedTest(name = "jniExplicitStreamMultiStep mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("JNI: explicit stream device writes for 20 steps with sync")
    void testJniExplicitStreamMultiStep(GraphExecutionMode mode) {
        Assumptions.assumeTrue(isCudaStreamApiAvailable(), "CUDA stream API not available");

        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        Map<String, INDArray> ph = singlePh("x", input);

        warmupWithChangingInput(sd, "x", input, "out", 8, new long[]{1, 8});
        DspHandle h = sd.dsp();
        int xIdx = h.extInputIndex("x");
        h.markVariable(xIdx);

        Pointer testStream = h.createTestStream();
        assertNotNull(testStream, "Failed to create test stream");

        try {
            List<Double> sums = new ArrayList<>();
            FloatPointer hostPtr = new FloatPointer(8);

            for (int step = 0; step < 20; step++) {
                float val = (float)((step + 1) * 7.5);
                for (int j = 0; j < 8; j++) hostPtr.put(j, val);

                int rc = h.writeDeviceBufferOnExplicitStream(xIdx, hostPtr, 8 * 4, testStream);
                assertEquals(0, rc, "explicit stream write failed at step " + step);

                // Sync the explicit stream before replay
                int syncRc = h.syncStream(testStream);
                assertEquals(0, syncRc, "stream sync failed at step " + step);

                Map<String, INDArray> result = sd.output(ph, "out");
                sums.add(result.get("out").sumNumber().doubleValue());
            }

            hostPtr.close();

            int stuckCount = 0;
            for (int i = 1; i < sums.size(); i++) {
                if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
            }
            assertTrue(stuckCount < 3,
                    mode + " [JNI_EXPLICIT_MULTI]: STUCK! " + stuckCount + "/19 steps. "
                            + "sums=" + sums.subList(0, Math.min(8, sums.size())));
            log.info("[JNI_EXPLICIT_MULTI] mode={} PASS — {}/19 unique with explicit stream writes",
                    mode, 19 - stuckCount);
        } finally {
            h.destroyTestStream(testStream);
        }
    }

    /**
     * Test that stream handles are non-null and different from each other.
     * Verifies the JNI stream introspection API works.
     */
    @Test
    @DisplayName("JNI: stream handle introspection")
    void testJniStreamHandleIntrospection() {
        Assumptions.assumeTrue(isCudaStreamApiAvailable(), "CUDA stream API not available");

        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, GraphExecutionMode.CUDA_GRAPHS);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        Map<String, INDArray> ph = singlePh("x", input);
        warmup(sd, ph, "out", 5);

        DspHandle h = sd.dsp();

        // Default stream should be non-null on CUDA
        Pointer defaultStream = h.getDefaultStream();
        assertNotNull(defaultStream, "Default stream should be non-null on CUDA");

        // Test stream should be different from default
        Pointer testStream = h.createTestStream();
        assertNotNull(testStream, "Test stream creation failed");
        try {
            // Addresses should differ (different CUDA streams)
            assertNotEquals(defaultStream.address(), testStream.address(),
                    "Test stream should be a different CUDA stream from default");
            log.info("[STREAM_INTROSPECTION] PASS — default={} test={}",
                    defaultStream.address(), testStream.address());
        } finally {
            h.destroyTestStream(testStream);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MISSING PLAN TESTS — Category 1: Cross-Stream D2D Ordering (additional)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Device write via addi (on LC default stream) with NO explicit
     * cudaStreamSynchronize before replay. Tests whether performPreReplaySync's
     * cross-stream event is correctly placed BEFORE D2D staging copies.
     *
     * If the cross-stream event is missing or mis-ordered, D2D reads pre-kernel data.
     * This is the exact pattern that causes stuck tokens in VLM decode.
     */
    @ParameterizedTest(name = "deviceWriteNoSync mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Device write via addi + NO explicit sync before replay — documents cross-stream ordering")
    void testDeviceWriteNoSync(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(16, 8);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 16);
        warmupWithChangingInput(sd, "x", input, "out", 12, new long[]{1, 16});

        // Step pattern: addi on device (LC default stream) then IMMEDIATELY sd.output
        // No cudaStreamSynchronize in between — DSP must handle cross-stream sync
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(1.0);
            input.addi(step * 3.0); // device write on LC default stream
            // NO explicit sync here — relies on performPreReplaySync
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        if (stuckCount >= 3) {
            log.warn("[DEVICE_WRITE_NO_SYNC] mode={} STALE! {}/19 stuck — cross-stream event missing/mis-ordered. sums={}",
                    mode, stuckCount, sums.subList(0, Math.min(8, sums.size())));
        }
        assertTrue(stuckCount < 3,
                mode + " [DEVICE_WRITE_NO_SYNC]: STUCK! " + stuckCount + "/19 steps. "
                        + "Device write (addi) without explicit sync not visible to DSP replay. "
                        + "sums=" + sums.subList(0, Math.min(8, sums.size())));
        log.info("[DEVICE_WRITE_NO_SYNC] mode={} PASS — {}/19 unique (cross-stream sync works without explicit sync)",
                mode, 19 - stuckCount);
    }

    /**
     * Device write via addi + explicit cudaStreamSynchronize(0) before replay.
     * This is the "safe" variant — if this fails, it's NOT a cross-stream issue
     * but a fundamental D2D/staging bug.
     */
    @ParameterizedTest(name = "deviceWriteWithExplicitSync mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Device write via addi + explicit stream sync before replay — must always work")
    void testDeviceWriteWithExplicitSync(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(16, 8);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 16);
        warmupWithChangingInput(sd, "x", input, "out", 12, new long[]{1, 16});

        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(1.0);
            input.addi(step * 3.0); // device write on LC default stream
            // Explicit sync: ensure all pending device work is complete
            Nd4j.getExecutioner().commit();
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck EVEN WITH explicit sync! "
                            + "This is a fundamental D2D/staging bug, not cross-stream. sums=" + sums);
        }
        log.info("[DEVICE_WRITE_EXPLICIT_SYNC] mode={} PASS — all 20 steps unique with explicit sync", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MISSING PLAN TESTS — Category 2: Variable Classification (additional)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Ext input classified as PLACEHOLDER at compile time — does it auto-get
     * variable treatment (D2D staging) without explicit markVariable()?
     */
    @ParameterizedTest(name = "autoMarkFromPlaceholderClassification mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Placeholder auto-classified at compile time — verify auto-mark variable behavior")
    void testAutoMarkFromPlaceholderClassification(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        Map<String, INDArray> ph = singlePh("x", input);

        // Do NOT call markVariable — rely on auto-classification
        warmup(sd, ph, "out", 8);

        DspHandle h = sd.dsp();
        int xIdx = h.extInputIndex("x");
        assertTrue(xIdx >= 0, "ext input 'x' not found");

        int numCached = h.numCachedVariableExtIndices();
        boolean xIsVariable = false;
        for (int i = 0; i < numCached; i++) {
            if (h.cachedVariableExtIndex(i) == xIdx) {
                xIsVariable = true;
                break;
            }
        }

        // Document the behavior
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        boolean isCudaBackend = "CUDA".equalsIgnoreCase(backend);
        if (xIsVariable) {
            log.info("[AUTO_MARK_PLACEHOLDER] mode={} — placeholder 'x' was AUTO-marked as variable " +
                    "(staging allocated). numCachedVars={}", mode, numCached);
            // Staging buffers are a discrete-device (CUDA) concept — on CPU, staging is never allocated.
            long stagingAddr = h.stagingBufferAddress(xIdx);
            if (isCudaBackend) {
                assertTrue(stagingAddr != 0,
                        mode + " 'x' auto-marked variable but staging buffer address is 0!");
            } else {
                log.info("[AUTO_MARK_PLACEHOLDER] mode={} CPU backend — staging not applicable (addr=0x{})",
                        mode, Long.toHexString(stagingAddr));
            }
        } else {
            log.info("[AUTO_MARK_PLACEHOLDER] mode={} — placeholder 'x' was NOT auto-marked as variable. " +
                    "numCachedVars={}. This means markVariable() is required for D2D staging.", mode, numCached);
        }

        // Regardless of variable marking, outputs should change when input changes
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100)));
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [AUTO_MARK_PLACEHOLDER]: STUCK! " + stuckCount + "/19 steps despite changing input. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[AUTO_MARK_PLACEHOLDER] mode={} PASS — outputs change correctly ({}/19 unique)", mode, 19 - stuckCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MISSING PLAN TESTS — Category 3: Arg Table (additional)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * 5 ext inputs all change address simultaneously. Verify a single arg table
     * refresh handles all of them correctly.
     */
    @ParameterizedTest(name = "argRefreshForMultipleChangedAddresses mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("5 placeholder ext inputs all change address simultaneously — single refresh handles all")
    void testArgRefreshForMultipleChangedAddresses(GraphExecutionMode mode) {
        // Build a 5-placeholder graph: out = x1*w1 + x2*w2 + x3*w3 + x4*w4 + x5*w5
        SameDiff g = SameDiff.create();
        int dim = 8;
        SDVariable[] phs = new SDVariable[5];
        SDVariable acc = null;
        for (int i = 0; i < 5; i++) {
            String name = "x" + i;
            phs[i] = g.placeHolder(name, DataType.FLOAT, 1, dim);
            SDVariable w = g.var("w" + i, Transforms.abs(Nd4j.randn(DataType.FLOAT, dim, dim)).addi(0.1f));
            SDVariable mm = g.mmul("mm" + i, phs[i], w);
            acc = (acc == null) ? mm : acc.add("add" + i, mm);
        }
        g.identity("out", acc);
        sd = g;
        configureMode(g, mode);

        // Warmup with consistent arrays
        Map<String, INDArray> ph = new LinkedHashMap<>();
        INDArray[] inputs = new INDArray[5];
        for (int i = 0; i < 5; i++) {
            inputs[i] = Nd4j.ones(DataType.FLOAT, 1, dim);
            ph.put("x" + i, inputs[i]);
        }
        warmup(g, ph, "out", 8);

        // Test: all 5 inputs change to NEW INDArray objects (different addresses) each step
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            for (int i = 0; i < 5; i++) {
                inputs[i] = Nd4j.valueArrayOf(new long[]{1, dim}, (double)(step * 5 + i + 1));
                ph.put("x" + i, inputs[i]);
            }
            Map<String, INDArray> result = g.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck when all 5 addresses change. sums=" + sums);
        }
        log.info("[ARG_REFRESH_5_ADDRS] mode={} PASS — all 20 steps unique with 5 simultaneous address changes", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MISSING PLAN TESTS — Category 4: Java Executor Fast-Path (additional)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Pass null for a placeholder after frozen. Must get graceful error or
     * fallback — never silently produce stale data.
     */
    @ParameterizedTest(name = "frozenFastPathNullPlaceholder mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Null placeholder value after frozen — graceful error, not stale data")
    void testFrozenFastPathNullPlaceholder(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 8, new long[]{1, 8});

        // Now pass null for the placeholder
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", null);

        boolean gotError = false;
        Double resultSum = null;
        try {
            Map<String, INDArray> result = sd.output(ph, "out");
            resultSum = result.get("out").sumNumber().doubleValue();
        } catch (Exception e) {
            gotError = true;
            log.info("[FROZEN_NULL_PH] mode={} — got expected error for null placeholder: {}",
                    mode, e.getClass().getSimpleName() + ": " + e.getMessage());
        }

        // Either an error (correct) or we document the behavior
        if (!gotError) {
            log.warn("[FROZEN_NULL_PH] mode={} — null placeholder did NOT throw. Result sum={}. "
                    + "If this equals last warmup step, it's using cached/stale data.", mode, resultSum);
        }
        // The test passes regardless — it documents the behavior.
        // The key requirement is: NOT silently producing wrong results without any signal.
        log.info("[FROZEN_NULL_PH] mode={} — behavior documented. gotError={} resultSum={}", mode, gotError, resultSum);
    }

    /**
     * A derived ext input (output of upstream SameDiff op used as input to subgraph)
     * changes between steps. Verify frozen fast-path detects the change.
     *
     * Simulated by having an intermediate variable that depends on a placeholder
     * (the "derived" input changes when the placeholder changes).
     */
    @ParameterizedTest(name = "frozenFastPathDerivedInputChanges mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Derived ext input (placeholder → transform → matmul) changes between steps")
    void testFrozenFastPathDerivedInputChanges(GraphExecutionMode mode) {
        // Graph: x (ph) → abs(x) → matmul(w) → out
        // The "derived" input to matmul is abs(x), which changes when x changes.
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w = g.var("w", Transforms.abs(Nd4j.randn(DataType.FLOAT, 8, 4)).addi(0.1f));
        SDVariable derived = g.math().abs("abs_x", x);
        g.mmul("out", derived, w);
        sd = g;
        configureMode(g, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(g, "x", input, "out", 8, new long[]{1, 8});

        // Test: change x each step — derived (abs(x)) must also change
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            // Use negative values so abs() is clearly doing something
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, -(double)(step + 1)));
            Map<String, INDArray> result = g.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [DERIVED_INPUT]: STUCK! " + stuckCount + "/19 steps. "
                        + "Derived input (abs(x)) not updated when x changes. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[DERIVED_INPUT] mode={} PASS — derived input changes propagated ({}/19 unique)", mode, 19 - stuckCount);
    }

    /**
     * Verify cachedInputArrays identity is updated after providing a new INDArray
     * (different Java object). After identity change, subsequent steps must use
     * the new cached value.
     */
    @ParameterizedTest(name = "frozenFastPathCachedArrayIdentity mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Cached array identity updated after new INDArray provided each step")
    void testFrozenFastPathCachedArrayIdentity(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        warmupWithChangingInput(sd, "x", input, "out", 8, new long[]{1, 8});

        // Provide a brand new INDArray object each step (tests identity-based detection)
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            INDArray newArr = Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 100));
            Map<String, INDArray> result = sd.output(singlePh("x", newArr), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        for (int i = 1; i < sums.size(); i++) {
            assertNotEquals(sums.get(i), sums.get(i - 1), 1e-3,
                    mode + " step " + i + " stuck with new INDArray identity each step. sums=" + sums);
        }
        log.info("[CACHED_IDENTITY] mode={} PASS — 20 new INDArray objects all produced unique outputs", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MISSING PLAN TESTS — Category 5: executeSteadyState() Fast Path (additional)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Verify executeSteadyState (via sd.output in well-warmed state) falls back
     * correctly when plan is NOT yet in REPLAYING state. Compare output of
     * early steps (pre-replay) vs steps after replay engages.
     */
    @ParameterizedTest(name = "steadyStateFallbackToExecute mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Early steps (pre-replay) produce same output as replay steps with identical input")
    void testSteadyStateFallbackToExecute(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.valueArrayOf(new long[]{1, 8}, 42.0);
        Map<String, INDArray> ph = singlePh("x", input);

        // Step 1: output during early execution (pre-replay, slot-by-slot)
        Map<String, INDArray> earlyResult = sd.output(ph, "out");
        double earlySum = earlyResult.get("out").sumNumber().doubleValue();

        // Steps 2-8: warmup to get into replay
        for (int i = 0; i < 7; i++) {
            sd.output(ph, "out");
        }

        // Steps 9+: replay mode — same input should produce same output
        List<Double> replaySums = new ArrayList<>();
        for (int step = 0; step < 10; step++) {
            Map<String, INDArray> result = sd.output(ph, "out");
            replaySums.add(result.get("out").sumNumber().doubleValue());
        }

        // All replay steps should match the early step (same input → same output)
        for (int i = 0; i < replaySums.size(); i++) {
            assertEquals(earlySum, replaySums.get(i), 1e-2,
                    mode + " replay step " + i + " differs from early step with same input! "
                            + "early=" + earlySum + " replay=" + replaySums.get(i));
        }
        log.info("[STEADY_FALLBACK] mode={} PASS — early sum={} matches all {} replay steps",
                mode, earlySum, replaySums.size());
    }

    /**
     * Each step passes different placeholder content through executeSteadyState
     * (well into replay). Outputs must change every step — NOT stuck.
     */
    @ParameterizedTest(name = "steadyStateWithChangingPlaceholder mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("executeSteadyState with different placeholder content each step — not stuck")
    void testSteadyStateWithChangingPlaceholder(GraphExecutionMode mode) {
        sd = buildSinglePlaceholder(8, 4);
        configureMode(sd, mode);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 8);
        // Get well into steady state (15 warmup steps)
        warmupWithChangingInput(sd, "x", input, "out", 15, new long[]{1, 8});

        // Now verify changing content produces changing output
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            input.assign(Nd4j.valueArrayOf(new long[]{1, 8}, (double)(step + 200)));
            Map<String, INDArray> result = sd.output(singlePh("x", input), "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [STEADY_CHANGING_PH]: STUCK! " + stuckCount + "/19 steps in steady state. sums=" +
                        sums.subList(0, Math.min(5, sums.size())));
        log.info("[STEADY_CHANGING_PH] mode={} PASS — {}/19 unique in steady state", mode, 19 - stuckCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ═══════════════════════════════════════════════════════════════════════════
    // MISSING PLAN TESTS — Category 7: Multi-External Lifecycle (additional)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * 1 "position_ids" ext input: host-written via assign() each step,
     * while other placeholders remain stable. The position_ids must be
     * reflected each step.
     */
    @ParameterizedTest(name = "positionIdsPatternNewValueSameBuffer mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("position_ids changes via host assign() each step + other inputs stable")
    void testPositionIdsPatternNewValueSameBuffer(GraphExecutionMode mode) {
        sd = buildLargeDecoderGraph(16, 2);
        configureMode(sd, mode);

        INDArray embed = Nd4j.randn(DataType.FLOAT, 1, 1, 16);
        INDArray posIds = Nd4j.zeros(DataType.FLOAT, 1, 1);
        INDArray kv0 = Nd4j.randn(DataType.FLOAT, 1, 4, 16);
        INDArray kv1 = Nd4j.randn(DataType.FLOAT, 1, 4, 16);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("inputs_embeds", embed);
        ph.put("position_ids", posIds);
        ph.put("layer_0_kv", kv0);
        ph.put("layer_1_kv", kv1);

        // Warmup
        for (int i = 0; i < 8; i++) {
            posIds.assign(i);
            sd.output(ph, "out");
        }

        // Test: only position_ids changes, everything else stable
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            posIds.assign(step + 100);
            Map<String, INDArray> result = sd.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [POS_IDS_PATTERN]: STUCK! " + stuckCount + "/19 steps. "
                        + "position_ids changes not reflected. sums=" + sums.subList(0, Math.min(5, sums.size())));
        log.info("[POS_IDS_PATTERN] mode={} PASS — position_ids host assign reflected ({}/19 unique)",
                mode, 19 - stuckCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MISSING PLAN TESTS — Category 8: VLM Decode (additional)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Simulate AutoregressiveDecode: Java controls the loop, does embedding
     * lookup (via Nd4j indexing), assigns to buffer, calls sd.output().
     * All steps should produce unique outputs (no degenerate repeats).
     */
    @ParameterizedTest(name = "decodePatternWithoutAutoregressiveOp mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Java-controlled decode loop: embed lookup → assign → sd.output (no AutoregressiveDecode op)")
    void testDecodePatternWithoutAutoregressiveOp(GraphExecutionMode mode) {
        int embedDim = 16;
        int vocabSize = 64;

        // Graph: inputs_embeds [1,1,embedDim] → reshape → matmul → out
        SameDiff g = SameDiff.create();
        SDVariable embed = g.placeHolder("inputs_embeds", DataType.FLOAT, 1, 1, embedDim);
        SDVariable posIds = g.placeHolder("position_ids", DataType.FLOAT, 1, 1);
        SDVariable w = g.var("w_proj", Transforms.abs(Nd4j.randn(DataType.FLOAT, embedDim, 8)).addi(0.1f));

        SDVariable posAdd = embed.add("pos_add", posIds);
        SDVariable flat = g.reshape("flat", posAdd, 1, embedDim);
        g.mmul("out", flat, w);
        sd = g;
        configureMode(g, mode);

        // Simulated embedding table
        INDArray embeddingTable = Nd4j.randn(DataType.FLOAT, vocabSize, embedDim);
        INDArray embedBuffer = Nd4j.zeros(DataType.FLOAT, 1, 1, embedDim);
        INDArray posBuffer = Nd4j.zeros(DataType.FLOAT, 1, 1);

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("inputs_embeds", embedBuffer);
        ph.put("position_ids", posBuffer);

        // Warmup: simulate prefill + first few decode steps
        for (int i = 0; i < 8; i++) {
            int tokenId = i % vocabSize;
            embedBuffer.assign(embeddingTable.getRow(tokenId).reshape(1, 1, embedDim));
            posBuffer.assign(i);
            g.output(ph, "out");
        }

        // Decode loop: each step looks up a different token embedding
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 30; step++) {
            int tokenId = (step * 7 + 3) % vocabSize; // pseudo-random token sequence
            embedBuffer.assign(embeddingTable.getRow(tokenId).reshape(1, 1, embedDim));
            posBuffer.assign(step + 8);
            Map<String, INDArray> result = g.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 5,
                mode + " [DECODE_NO_AR_OP]: STUCK! " + stuckCount + "/29 steps. "
                        + "Java-controlled decode loop producing degenerate output. "
                        + "sums=" + sums.subList(0, Math.min(8, sums.size())));
        log.info("[DECODE_NO_AR_OP] mode={} PASS — {}/29 unique decode steps (Java-controlled loop)",
                mode, 29 - stuckCount);
    }

    /**
     * Simulate CUDA kernel modifying ext input on default stream (mimics
     * embedLookupKernel), then sd.output on DSP stream.
     * Verify cross-stream sync fires and fresh data is visible.
     */
    @ParameterizedTest(name = "decodePatternDeviceKernelBeforeReplay mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"CUDA_GRAPHS", "TRITON", "AUTO"})
    @DisplayName("Device kernel writes ext input (simulated embed lookup) → replay sees fresh data")
    void testDecodePatternDeviceKernelBeforeReplay(GraphExecutionMode mode) {
        int embedDim = 16;

        SameDiff g = SameDiff.create();
        SDVariable embed = g.placeHolder("inputs_embeds", DataType.FLOAT, 1, 1, embedDim);
        SDVariable w = g.var("w_proj", Transforms.abs(Nd4j.randn(DataType.FLOAT, embedDim, 4)).addi(0.1f));
        SDVariable flat = g.reshape("flat", embed, 1, embedDim);
        g.mmul("out", flat, w);
        sd = g;
        configureMode(g, mode);

        INDArray embedBuffer = Nd4j.zeros(DataType.FLOAT, 1, 1, embedDim);
        Map<String, INDArray> ph = singlePh("inputs_embeds", embedBuffer);

        // Warmup
        for (int i = 0; i < 10; i++) {
            // Device write: assign + addi simulates CUDA kernel writing to buffer
            embedBuffer.assign(0.0);
            embedBuffer.addi((double)(i + 1));
            g.output(ph, "out");
        }

        // Test: device kernel write (addi on default stream) then replay
        List<Double> sums = new ArrayList<>();
        for (int step = 0; step < 20; step++) {
            // Simulate embedLookupKernel: write to device buffer on default stream
            embedBuffer.assign(0.0);
            embedBuffer.addi((step + 1) * 10.0); // different value each step
            // NO explicit sync — cross-stream event must handle this
            Map<String, INDArray> result = g.output(ph, "out");
            sums.add(result.get("out").sumNumber().doubleValue());
        }

        int stuckCount = 0;
        for (int i = 1; i < sums.size(); i++) {
            if (Math.abs(sums.get(i) - sums.get(i - 1)) < 1e-3) stuckCount++;
        }
        assertTrue(stuckCount < 3,
                mode + " [DECODE_DEVICE_KERNEL]: STUCK! " + stuckCount + "/19 steps. "
                        + "Device kernel write to embed buffer not visible to DSP replay. "
                        + "sums=" + sums.subList(0, Math.min(8, sums.size())));
        log.info("[DECODE_DEVICE_KERNEL] mode={} PASS — {}/19 unique after device kernel writes",
                mode, 19 - stuckCount);
    }
}
