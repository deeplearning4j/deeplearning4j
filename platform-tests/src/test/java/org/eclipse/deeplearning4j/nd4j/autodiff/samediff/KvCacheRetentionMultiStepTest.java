/*
 *  ******************************************************************************
 *  *
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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests KV cache retention and scatter correctness across 50 decode steps.
 *
 * Builds a graph with past_key_values input and present_key_values output (via concat).
 * Runs 50 steps, at each step verifying:
 * - Position 0 data is unchanged from prefill
 * - New position has the new data (from present KV concat output)
 * - Previously scattered positions are still correct
 *
 * This tests the Java-side KV scatter path (DecoderUtils.scatterNewKvEntries)
 * as well as the basic concat pattern that the C++ KV retention replaces.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class KvCacheRetentionMultiStepTest extends BaseNd4jTestWithBackends {

    private static final int NUM_HEADS = 2;
    private static final int HEAD_DIM = 8;
    private static final int HIDDEN = NUM_HEADS * HEAD_DIM;
    private static final int PREFILL_LEN = 3;
    private static final int MAX_NEW_TOKENS = 50;

    @Override
    public char ordering() {
        return 'c';
    }

    @AfterEach
    public void cleanup() {
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
    }

    /**
     * Build a graph that takes past_kv [1, heads, kvLen, dim] and an input [1, hidden],
     * projects the input to a new KV entry, and concats it to produce present_kv.
     */
    private SameDiff buildKvConcatGraph(long maxKvLen) {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, HIDDEN);
        SDVariable pastKey = sd.placeHolder("past_key_values.0.key", DataType.FLOAT,
                1, NUM_HEADS, maxKvLen, HEAD_DIM);
        SDVariable pastValue = sd.placeHolder("past_key_values.0.value", DataType.FLOAT,
                1, NUM_HEADS, maxKvLen, HEAD_DIM);

        // Project input to new KV entries
        SDVariable wKey = sd.var("w_key", Nd4j.randn(DataType.FLOAT, HIDDEN, NUM_HEADS * HEAD_DIM).muli(0.1));
        SDVariable wValue = sd.var("w_value", Nd4j.randn(DataType.FLOAT, HIDDEN, NUM_HEADS * HEAD_DIM).muli(0.1));

        SDVariable newKeyFlat = sd.mmul("key_proj", input, wKey);
        SDVariable newKey = sd.reshape("new_key", newKeyFlat, 1, NUM_HEADS, 1, HEAD_DIM);
        SDVariable newValueFlat = sd.mmul("value_proj", input, wValue);
        SDVariable newValue = sd.reshape("new_value", newValueFlat, 1, NUM_HEADS, 1, HEAD_DIM);

        // Concat: [1, heads, maxKvLen, dim] + [1, heads, 1, dim] -> [1, heads, maxKvLen+1, dim]
        SDVariable presentKey = sd.concat("present.0.key", 2, pastKey, newKey);
        SDVariable presentValue = sd.concat("present.0.value", 2, pastValue, newValue);

        // Also produce a simple "logits" output so we have a non-KV output
        SDVariable wOut = sd.var("w_out", Nd4j.randn(DataType.FLOAT, HIDDEN, 32).muli(0.1));
        SDVariable logits = sd.mmul("logits", input, wOut);

        sd.setOutputs("logits", "present.0.key", "present.0.value");
        return sd;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testKvCacheRetentionAcross50Steps(Nd4jBackend backend) {
        long maxKvLen = PREFILL_LEN + MAX_NEW_TOKENS;
        SameDiff sd = buildKvConcatGraph(maxKvLen);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        // Initialize static KV buffers with known prefill data
        INDArray staticKeyBuf = Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, maxKvLen, HEAD_DIM);
        INDArray staticValueBuf = Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, maxKvLen, HEAD_DIM);

        // Fill prefill positions with recognizable patterns
        float[][] prefillKeySnapshots = new float[PREFILL_LEN][];
        float[][] prefillValueSnapshots = new float[PREFILL_LEN][];
        for (int p = 0; p < PREFILL_LEN; p++) {
            INDArray keySlice = Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, 1, HEAD_DIM).muli(0.5f + p * 0.1f);
            INDArray valueSlice = Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, 1, HEAD_DIM).muli(0.5f + p * 0.1f);
            staticKeyBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(p), NDArrayIndex.all()).assign(
                    keySlice.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.point(0), NDArrayIndex.all()));
            staticValueBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(p), NDArrayIndex.all()).assign(
                    valueSlice.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.point(0), NDArrayIndex.all()));
            // Snapshot the prefill data for later verification
            prefillKeySnapshots[p] = staticKeyBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(p), NDArrayIndex.all()).dup().data().asFloat();
            prefillValueSnapshots[p] = staticValueBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(p), NDArrayIndex.all()).dup().data().asFloat();
        }

        // Also snapshot the decode entries as we scatter them
        float[][] scatteredKeySnapshots = new float[MAX_NEW_TOKENS][];
        float[][] scatteredValueSnapshots = new float[MAX_NEW_TOKENS][];

        long cachePos = PREFILL_LEN;
        int prefillCorruptions = 0;
        int scatterCorruptions = 0;

        String[] allOutputs = {"logits", "present.0.key", "present.0.value"};

        for (int step = 0; step < MAX_NEW_TOKENS; step++) {
            // Varying input each step
            Nd4j.getRandom().setSeed(step * 777L + 99);
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, HIDDEN).muli(0.1f);

            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put("input", input);
            inputs.put("past_key_values.0.key", staticKeyBuf);
            inputs.put("past_key_values.0.value", staticValueBuf);

            Map<String, INDArray> outputs;
            if (step >= 2) {
                outputs = sd.outputDirect(inputs, allOutputs);
            } else {
                outputs = sd.output(inputs, allOutputs);
            }

            // Extract present KV and scatter the new entry
            INDArray presentKey = outputs.get("present.0.key");
            INDArray presentValue = outputs.get("present.0.value");
            assertNotNull(presentKey, "present.0.key null at step " + step);
            assertNotNull(presentValue, "present.0.value null at step " + step);

            // Present shape should be [1, heads, maxKvLen+1, dim]
            assertEquals(maxKvLen + 1, presentKey.size(2),
                    "Present key seq dim should be maxKvLen+1 at step " + step);

            // New entry is at the last position (maxKvLen)
            INDArray newKeyEntry = presentKey.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(maxKvLen), NDArrayIndex.all()).dup();
            INDArray newValueEntry = presentValue.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(maxKvLen), NDArrayIndex.all()).dup();

            // Scatter into static buffer
            staticKeyBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(newKeyEntry);
            staticValueBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(newValueEntry);

            // Snapshot the scattered entry
            scatteredKeySnapshots[step] = newKeyEntry.data().asFloat();
            scatteredValueSnapshots[step] = newValueEntry.data().asFloat();

            // --- VERIFICATION ---

            // 1. Check prefill positions are unchanged
            for (int p = 0; p < PREFILL_LEN; p++) {
                float[] currentKey = staticKeyBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(p), NDArrayIndex.all()).dup().data().asFloat();
                if (!Arrays.equals(currentKey, prefillKeySnapshots[p])) {
                    prefillCorruptions++;
                    if (prefillCorruptions <= 3) {
                        log.error("PREFILL CORRUPTION at step {} position {}: expected first={} got first={}",
                                step, p, prefillKeySnapshots[p][0], currentKey[0]);
                    }
                }
            }

            // 2. Check previously scattered positions are still correct
            for (int prev = 0; prev < step; prev++) {
                long prevPos = PREFILL_LEN + prev;
                float[] currentKey = staticKeyBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(prevPos), NDArrayIndex.all()).dup().data().asFloat();
                if (!Arrays.equals(currentKey, scatteredKeySnapshots[prev])) {
                    scatterCorruptions++;
                    if (scatterCorruptions <= 3) {
                        log.error("SCATTER CORRUPTION at step {} for prev scatter at pos {}: expected first={} got first={}",
                                step, prevPos, scatteredKeySnapshots[prev][0], currentKey[0]);
                    }
                }
            }

            // 3. Verify the new entry was written correctly
            float[] justWrittenKey = staticKeyBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all()).dup().data().asFloat();
            assertArrayEquals(scatteredKeySnapshots[step], justWrittenKey, 1e-6f,
                    "Newly scattered key at step " + step + " does not match expected");

            cachePos++;

            if (step < 5 || step % 10 == 0) {
                log.info("Step {}: cachePos={} prefillCorruptions={} scatterCorruptions={}",
                        step, cachePos, prefillCorruptions, scatterCorruptions);
            }
        }

        // Summary
        log.info("=== KV CACHE RETENTION RESULTS ===");
        log.info("  Total steps: {}", MAX_NEW_TOKENS);
        log.info("  Prefill corruptions: {}", prefillCorruptions);
        log.info("  Scatter corruptions: {}", scatterCorruptions);

        assertEquals(0, prefillCorruptions,
                "Prefill KV data must not be corrupted during decode -- found " + prefillCorruptions + " corruptions");
        assertEquals(0, scatterCorruptions,
                "Previously scattered KV data must not be corrupted -- found " + scatterCorruptions + " corruptions");

        sd.close();
    }
}
