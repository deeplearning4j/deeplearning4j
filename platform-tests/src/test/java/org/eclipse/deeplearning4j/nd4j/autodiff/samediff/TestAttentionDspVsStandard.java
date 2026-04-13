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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Minimal test reproducing DSP vs standard execution divergence for attention.
 *
 * Bug: onnx_multi_head_attention produces different output in DSP execution vs
 * standard execution. Divergence observed at maxDiff=0.56 on SmolDocling decoder
 * (batch=1, seqLen=17, hidden=576, numHeads=9, headDim=64).
 *
 * This test builds a small SameDiff graph programmatically (no model loading) to
 * reproduce the divergence in seconds.
 *
 * Parameters matching SmolDocling:
 *   batch=1, seqLen=17, hidden=576, numHeads=9, headDim=64
 *   useCausalMask=1, scale=0 (auto = 1/sqrt(64) = 0.125)
 *
 * Run:
 *   cd platform-tests && mvn test \
 *     -Dtest=TestAttentionDspVsStandard \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TestAttentionDspVsStandard {

    private static final double TOLERANCE = 1e-4;
    private static boolean tritonAvailable;

    @BeforeAll
    public static void setup() {
        tritonAvailable = Nd4j.getNativeOps().isTritonAvailable();
        log.info("Triton available: {}", tritonAvailable);
        Nd4j.getRandom().setSeed(12345);
    }

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    // ─── Core test: DSP vs standard for Q/K/V → MHA ────────────────────────

    @Test
    @DisplayName("DSP vs standard: Q/K/V projections → onnx_multi_head_attention matches")
    public void testAttentionDspVsStandard() {
        // SmolDocling parameters
        int batch = 1;
        int seqLen = 17;
        int hidden = 576;
        int numHeads = 9;
        int headDim = 64; // hidden / numHeads = 576 / 9 = 64

        log.info("Building attention graph: batch={}, seqLen={}, hidden={}, heads={}, headDim={}",
                batch, seqLen, hidden, numHeads, headDim);

        SameDiff sd = SameDiff.create();

        // Fixed seed for reproducibility
        Nd4j.getRandom().setSeed(12345);

        // Input placeholder: [1, seqLen, hidden]
        SDVariable inputEmbeds = sd.placeHolder("input_embeds", DataType.FLOAT, batch, seqLen, hidden);

        // Weight matrices: [hidden, hidden] — constants
        INDArray wq = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray wk = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray wv = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));

        SDVariable wqVar = sd.constant("W_q", wq);
        SDVariable wkVar = sd.constant("W_k", wk);
        SDVariable wvVar = sd.constant("W_v", wv);

        // Q/K/V projections: matmul(input, W) → [1, seqLen, hidden]
        SDVariable q = sd.mmul("q_proj", inputEmbeds, wqVar);
        SDVariable k = sd.mmul("k_proj", inputEmbeds, wkVar);
        SDVariable v = sd.mmul("v_proj", inputEmbeds, wvVar);

        // Causal mask: [1, numHeads, seqLen, seqLen]
        // 0 for allowed (lower-triangular), -3.4e38 for masked (upper-triangular above diagonal)
        INDArray causalMask = Nd4j.ones(DataType.FLOAT, batch, numHeads, seqLen, seqLen);
        for (int h = 0; h < numHeads; h++) {
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    if (j > i) {
                        // Future token: mask with -inf
                        causalMask.putScalar(new int[]{0, h, i, j}, -3.4028235e+38f);
                    } else {
                        causalMask.putScalar(new int[]{0, h, i, j}, 0.0f);
                    }
                }
            }
        }
        SDVariable attnBias = sd.constant("attn_bias", causalMask);

        // onnx_multi_head_attention op
        // Inputs: query, key, value, attention_bias, past_key, past_value
        // Int args: numHeads, useCausalMask
        // Float args: scale (0 = auto)
        OnnxMultiHeadAttention mhaOp = new OnnxMultiHeadAttention(
                sd, q, k, v, attnBias, null, null,
                numHeads, 0.0, false, 1);
        SDVariable[] mhaOutputs = mhaOp.outputVariables();

        SDVariable attentionOutput = mhaOutputs[0];
        String autoName = attentionOutput.name();
        sd.renameVariable(autoName, "attention_output");

        // Create input data
        Nd4j.getRandom().setSeed(12345);
        INDArray inputData = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("input_embeds", inputData.dup());

        List<String> outputsToCompare = Collections.singletonList("attention_output");

        // Run compareExecutionPaths: DSP (SLOT_BY_SLOT) vs standard
        log.info("Running compareExecutionPaths with tolerance={}", TOLERANCE);
        Map<String, Double> divergences = sd.compareExecutionPaths(
                feedDict, outputsToCompare, TOLERANCE, GraphExecutionMode.SLOT_BY_SLOT);

        if (divergences.isEmpty()) {
            log.info("No divergence detected between DSP and standard execution");
        } else {
            for (Map.Entry<String, Double> entry : divergences.entrySet()) {
                log.error("DIVERGENCE [{}]: maxDiff={}", entry.getKey(), entry.getValue());
            }
            fail("DSP vs standard execution diverged. Divergences: " + divergences);
        }

        sd.close();
    }

    // ─── Variant: with attention mask (more realistic to SmolDocling) ──────

    @Test
    @DisplayName("DSP vs standard: attention with explicit causal mask (no useCausalMask flag)")
    public void testAttentionWithExplicitCausalMask() {
        int batch = 1;
        int seqLen = 17;
        int hidden = 576;
        int numHeads = 9;

        log.info("Building attention graph with explicit causal mask");

        SameDiff sd = SameDiff.create();
        Nd4j.getRandom().setSeed(12345);

        SDVariable inputEmbeds = sd.placeHolder("input_embeds", DataType.FLOAT, batch, seqLen, hidden);

        INDArray wq = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray wk = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray wv = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));

        SDVariable wqVar = sd.constant("W_q", wq);
        SDVariable wkVar = sd.constant("W_k", wk);
        SDVariable wvVar = sd.constant("W_v", wv);

        SDVariable q = sd.mmul("q_proj", inputEmbeds, wqVar);
        SDVariable k = sd.mmul("k_proj", inputEmbeds, wkVar);
        SDVariable v = sd.mmul("v_proj", inputEmbeds, wvVar);

        // Explicit causal mask as 4D tensor
        INDArray causalMask = createCausalMask(batch, numHeads, seqLen);
        SDVariable attnBias = sd.constant("attn_bias", causalMask);

        OnnxMultiHeadAttention mhaOp = new OnnxMultiHeadAttention(
                sd, q, k, v, attnBias, null, null,
                numHeads, 0.0, false, 1);
        SDVariable[] mhaOutputs = mhaOp.outputVariables();

        SDVariable attentionOutput = mhaOutputs[0];
        String autoName = attentionOutput.name();
        sd.renameVariable(autoName, "attention_output");

        Nd4j.getRandom().setSeed(12345);
        INDArray inputData = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("input_embeds", inputData.dup());

        List<String> outputsToCompare = Collections.singletonList("attention_output");

        log.info("Running compareExecutionPaths with explicit causal mask");
        Map<String, Double> divergences = sd.compareExecutionPaths(
                feedDict, outputsToCompare, TOLERANCE, GraphExecutionMode.SLOT_BY_SLOT);

        if (!divergences.isEmpty()) {
            for (Map.Entry<String, Double> entry : divergences.entrySet()) {
                log.error("DIVERGENCE [{}]: maxDiff={}", entry.getKey(), entry.getValue());
            }
            fail("DSP vs standard execution diverged. Divergences: " + divergences);
        }

        sd.close();
    }

    // ─── Variant: simpler — just MHA op without projections ────────────────

    @Test
    @DisplayName("DSP vs standard: bare onnx_multi_head_attention (no projections)")
    public void testBareMhaDspVsStandard() {
        int batch = 1;
        int seqLen = 17;
        int numHeads = 9;
        int headDim = 64;
        int hidden = numHeads * headDim;

        log.info("Building bare MHA graph: Q/K/V directly as placeholders");

        SameDiff sd = SameDiff.create();
        Nd4j.getRandom().setSeed(12345);

        SDVariable q = sd.placeHolder("query", DataType.FLOAT, batch, seqLen, hidden);
        SDVariable k = sd.placeHolder("key", DataType.FLOAT, batch, seqLen, hidden);
        SDVariable v = sd.placeHolder("value", DataType.FLOAT, batch, seqLen, hidden);

        // No attention bias (null), no past KV
        OnnxMultiHeadAttention mhaOp = new OnnxMultiHeadAttention(
                sd, q, k, v, null, null, null,
                numHeads, 0.0, true, 1);
        SDVariable[] mhaOutputs = mhaOp.outputVariables();

        SDVariable attentionOutput = mhaOutputs[0];
        String autoName = attentionOutput.name();
        sd.renameVariable(autoName, "attention_output");

        Nd4j.getRandom().setSeed(12345);
        INDArray qData = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray kData = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray vData = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("query", qData.dup());
        feedDict.put("key", kData.dup());
        feedDict.put("value", vData.dup());

        List<String> outputsToCompare = Collections.singletonList("attention_output");

        log.info("Running compareExecutionPaths for bare MHA");
        Map<String, Double> divergences = sd.compareExecutionPaths(
                feedDict, outputsToCompare, TOLERANCE, GraphExecutionMode.SLOT_BY_SLOT);

        if (!divergences.isEmpty()) {
            for (Map.Entry<String, Double> entry : divergences.entrySet()) {
                log.error("DIVERGENCE [{}]: maxDiff={}", entry.getKey(), entry.getValue());
            }
            fail("DSP vs standard execution diverged. Divergences: " + divergences);
        }

        sd.close();
    }

    // ─── Helper methods ─────────────────────────────────────────────────────

    /**
     * Creates a causal attention mask: lower-triangular with 0 for allowed,
     * -3.4e38 for masked future positions.
     */
    private static INDArray createCausalMask(int batch, int numHeads, int seqLen) {
        INDArray mask = Nd4j.ones(DataType.FLOAT, batch, numHeads, seqLen, seqLen);
        for (int h = 0; h < numHeads; h++) {
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    if (j > i) {
                        mask.putScalar(new int[]{0, h, i, j}, -3.4028235e+38f);
                    } else {
                        mask.putScalar(new int[]{0, h, i, j}, 0.0f);
                    }
                }
            }
        }
        return mask;
    }
}
