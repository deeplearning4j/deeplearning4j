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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
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
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Multi-step DSP decode test that mimics the real decoder pattern:
 * - Inputs: input_ids [1,1], position_ids [1,1], attention_mask [1,N], past_key_values [1,heads,maxLen,dim]
 * - Ops: matmul + add (bias) + activation, KV concat pattern
 * - Outputs: logits [1,1,vocab], present_key_values [1,heads,maxLen+1,dim]
 *
 * Runs 50 decode steps transitioning from output() to outputDirect() with frozen shapes.
 * Detects progressive degeneration where outputs stop changing despite different inputs.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class MultiStepDspDecodeTest extends BaseNd4jTestWithBackends {

    private static final int NUM_HEADS = 4;
    private static final int HEAD_DIM = 16;
    private static final int HIDDEN_SIZE = NUM_HEADS * HEAD_DIM; // 64
    private static final int VOCAB_SIZE = 128;
    private static final int MAX_NEW_TOKENS = 50;
    private static final int PREFILL_LEN = 5;

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
     * Build a synthetic "decoder" SameDiff graph that mimics a transformer decoder layer.
     *
     * Graph pattern:
     *   inputs_embeds [1, seqLen, hidden] --> matmul(Wq) --> reshape to [1, heads, seqLen, headDim]
     *   past_key_values.0.key [1, heads, kvLen, headDim] --> concat with new key --> present.0.key
     *   attention scores = Q * K^T --> add(attention_mask) --> softmax --> matmul(V) --> project --> logits
     *
     * Simplified to: inputs_embeds -> matmul -> add -> tanh -> matmul(vocab_proj) -> logits
     * Plus: past_kv concat with projected input to produce present_kv
     */
    private SameDiff buildSyntheticDecoder(long maxKvLen) {
        SameDiff sd = SameDiff.create();

        // Placeholders
        SDVariable inputsEmbeds = sd.placeHolder("inputs_embeds", DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        SDVariable inputIds = sd.placeHolder("input_ids", DataType.LONG, 1, 1);
        SDVariable positionIds = sd.placeHolder("position_ids", DataType.LONG, 1, 1);
        SDVariable attentionMask = sd.placeHolder("attention_mask", DataType.LONG, 1, maxKvLen + 1);
        SDVariable pastKey = sd.placeHolder("past_key_values.0.key", DataType.FLOAT, 1, NUM_HEADS, maxKvLen, HEAD_DIM);
        SDVariable pastValue = sd.placeHolder("past_key_values.0.value", DataType.FLOAT, 1, NUM_HEADS, maxKvLen, HEAD_DIM);

        // Weights
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, HIDDEN_SIZE, HIDDEN_SIZE).muli(0.02));
        SDVariable b1 = sd.var("b1", Nd4j.zeros(DataType.FLOAT, HIDDEN_SIZE));
        SDVariable wVocab = sd.var("w_vocab", Nd4j.randn(DataType.FLOAT, HIDDEN_SIZE, VOCAB_SIZE).muli(0.02));
        SDVariable wKey = sd.var("w_key", Nd4j.randn(DataType.FLOAT, HIDDEN_SIZE, NUM_HEADS * HEAD_DIM).muli(0.02));
        SDVariable wValue = sd.var("w_value", Nd4j.randn(DataType.FLOAT, HIDDEN_SIZE, NUM_HEADS * HEAD_DIM).muli(0.02));

        // Forward pass: inputs_embeds [1,1,hidden] -> squeeze -> [1, hidden]
        SDVariable squeezed = sd.squeeze("squeeze_embed", inputsEmbeds, 1);  // [1, hidden]

        // Layer 1: matmul + bias + tanh
        SDVariable h1 = sd.mmul("h1_matmul", squeezed, w1);          // [1, hidden]
        h1 = h1.add("h1_add", b1);                                    // [1, hidden]
        h1 = sd.math.tanh("h1_act", h1);                             // [1, hidden]

        // Project to new KV entries: [1, hidden] -> [1, heads*headDim] -> reshape [1, heads, 1, headDim]
        SDVariable newKeyFlat = sd.mmul("new_key_proj", h1, wKey);    // [1, heads*headDim]
        SDVariable newKey = sd.reshape("new_key_reshape", newKeyFlat, 1, NUM_HEADS, 1, HEAD_DIM);
        SDVariable newValueFlat = sd.mmul("new_value_proj", h1, wValue);
        SDVariable newValue = sd.reshape("new_value_reshape", newValueFlat, 1, NUM_HEADS, 1, HEAD_DIM);

        // Concat past KV with new KV: [1, heads, maxLen, dim] concat [1, heads, 1, dim] -> [1, heads, maxLen+1, dim]
        SDVariable presentKey = sd.concat("present.0.key", 2, pastKey, newKey);
        SDVariable presentValue = sd.concat("present.0.value", 2, pastValue, newValue);

        // Logits projection: [1, hidden] -> [1, vocab] -> reshape [1, 1, vocab]
        SDVariable logitsFlat = sd.mmul("logits_proj", h1, wVocab);   // [1, vocab]
        SDVariable logits = sd.reshape("logits", logitsFlat, 1, 1, VOCAB_SIZE);

        // Mark outputs
        sd.setOutputs("logits", "present.0.key", "present.0.value");

        return sd;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiStepDecodeWithDsp(Nd4jBackend backend) {
        long maxKvLen = PREFILL_LEN + MAX_NEW_TOKENS;
        SameDiff sd = buildSyntheticDecoder(maxKvLen);

        // Enable DSP
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        // Initialize static KV buffers (simulating post-prefill state)
        INDArray staticKeyBuf = Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, maxKvLen, HEAD_DIM).muli(0.1f);
        INDArray staticValueBuf = Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, maxKvLen, HEAD_DIM).muli(0.1f);
        // Fill first PREFILL_LEN positions with "prefill data"
        for (int p = PREFILL_LEN; p < maxKvLen; p++) {
            staticKeyBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(p), NDArrayIndex.all()).assign(0);
            staticValueBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(p), NDArrayIndex.all()).assign(0);
        }
        Nd4j.getExecutioner().commit();

        // Reusable decode buffers (fixed address for CUDA graph replay)
        INDArray reusableEmbeds = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE);
        INDArray reusableInputIds = Nd4j.createFromArray(new long[][]{{1}});
        INDArray reusablePositionIds = Nd4j.createFromArray(new long[][]{{PREFILL_LEN}});
        INDArray attentionMask = Nd4j.zeros(DataType.LONG, 1, maxKvLen + 1);
        // Mark prefill positions as attended
        attentionMask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, PREFILL_LEN)).assign(1);
        Nd4j.getExecutioner().commit();

        long cachePos = PREFILL_LEN;
        Set<Integer> uniqueArgmaxTokens = new HashSet<>();
        Map<Integer, float[]> outputsByStep = new LinkedHashMap<>();
        int firstRepeatStep = -1;
        float[] prevLogits = null;

        String[] allOutputs = {"logits", "present.0.key", "present.0.value"};

        for (int step = 0; step < MAX_NEW_TOKENS; step++) {
            // Vary input each step: different "token embedding"
            int tokenId = (step * 7 + 13) % VOCAB_SIZE;  // deterministic but varying
            reusableEmbeds.assign(Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN_SIZE).muli(0.1f));
            reusableInputIds.putScalar(0, 0, tokenId);
            reusablePositionIds.putScalar(0, 0, cachePos);
            attentionMask.putScalar(0, cachePos, 1);
            // Also set concat position
            attentionMask.putScalar(0, maxKvLen, 1);
            Nd4j.getExecutioner().commit();

            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put("inputs_embeds", reusableEmbeds);
            inputs.put("input_ids", reusableInputIds);
            inputs.put("position_ids", reusablePositionIds);
            inputs.put("attention_mask", attentionMask);
            inputs.put("past_key_values.0.key", staticKeyBuf);
            inputs.put("past_key_values.0.value", staticValueBuf);

            Map<String, INDArray> outputs;
            boolean useDirect = step >= 2;
            if (useDirect) {
                // Freeze shapes on first direct call
                if (step == 2) {
                    InferenceSession session = sd.getOrCreateSession();
                    DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
                    if (dspExec != null) {
                        dspExec.setShapesFrozen(true);
                        log.info("Step {}: Shapes frozen for CUDA graph capture", step);
                    }
                }
                outputs = sd.outputDirect(inputs, allOutputs);
            } else {
                outputs = sd.output(inputs, allOutputs);
            }
            Nd4j.getExecutioner().commit();

            // Extract logits
            INDArray logitsArr = outputs.get("logits");
            assertNotNull(logitsArr, "Logits null at step " + step);
            assertEquals(3, logitsArr.rank(), "Logits should be rank 3 at step " + step);

            // Get argmax token
            float[] logitsData = logitsArr.dup().data().asFloat();
            int argmax = 0;
            float maxVal = logitsData[0];
            for (int i = 1; i < logitsData.length; i++) {
                if (logitsData[i] > maxVal) {
                    maxVal = logitsData[i];
                    argmax = i;
                }
            }
            uniqueArgmaxTokens.add(argmax);
            outputsByStep.put(step, Arrays.copyOf(logitsData, Math.min(10, logitsData.length)));

            // Check for stale/repeated outputs
            if (prevLogits != null && firstRepeatStep < 0) {
                boolean identical = true;
                for (int i = 0; i < Math.min(prevLogits.length, logitsData.length); i++) {
                    if (Math.abs(prevLogits[i] - logitsData[i]) > 1e-6f) {
                        identical = false;
                        break;
                    }
                }
                if (identical) {
                    firstRepeatStep = step;
                    log.warn("DEGENERATION DETECTED: Step {} output identical to step {}", step, step - 1);
                }
            }
            prevLogits = logitsData;

            // Scatter new KV entry into static buffer
            INDArray presentKey = outputs.get("present.0.key");
            INDArray presentValue = outputs.get("present.0.value");
            if (presentKey != null && presentValue != null) {
                // New entry is at position maxKvLen (the concat'd position)
                INDArray newKeyEntry = presentKey.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(maxKvLen), NDArrayIndex.all());
                INDArray newValueEntry = presentValue.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(maxKvLen), NDArrayIndex.all());
                // Write into static buffer at cachePos
                staticKeyBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(newKeyEntry);
                staticValueBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(newValueEntry);
                Nd4j.getExecutioner().commit();
            }

            cachePos++;

            if (step < 5 || step % 10 == 0) {
                log.info("Step {}: argmax={} maxVal={} unique_so_far={} useDirect={}",
                        step, argmax, maxVal, uniqueArgmaxTokens.size(), useDirect);
            }
        }

        // Summary
        double diversity = (double) uniqueArgmaxTokens.size() / MAX_NEW_TOKENS;
        log.info("=== MULTI-STEP DSP DECODE RESULTS ===");
        log.info("  Total steps: {}", MAX_NEW_TOKENS);
        log.info("  Unique argmax tokens: {} / {} ({}%)", uniqueArgmaxTokens.size(),
                MAX_NEW_TOKENS, String.format("%.1f", diversity * 100));
        log.info("  First repeat step: {}", firstRepeatStep);

        // Key assertions
        assertTrue(uniqueArgmaxTokens.size() > 1,
                "Outputs must not all be the same token -- got only " + uniqueArgmaxTokens.size() + " unique");
        assertTrue(diversity > 0.10,
                "Output diversity must be >10% -- got " + (diversity * 100) + "% (" +
                        uniqueArgmaxTokens.size() + "/" + MAX_NEW_TOKENS + " unique)");
        if (firstRepeatStep >= 0) {
            log.warn("Degeneration started at step {} -- outputs became identical to previous step", firstRepeatStep);
            // Allow some repetition but flag if it starts too early
            assertTrue(firstRepeatStep > 10,
                    "Degeneration should not start before step 10 -- started at step " + firstRepeatStep);
        }

        sd.close();
    }
}
