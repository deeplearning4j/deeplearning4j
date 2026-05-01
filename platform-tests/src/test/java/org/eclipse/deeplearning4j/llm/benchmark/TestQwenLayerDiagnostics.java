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

package org.eclipse.deeplearning4j.llm.benchmark;

import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.DownloadResult;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.LLMModel;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.QuantType;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Layer-by-layer diagnostic test for Qwen3.5-0.8B forward pass correctness.
 *
 * <p>The model produces garbage/random tokens in inference. This test performs a single
 * prefill forward pass and captures intermediate hidden states at each major checkpoint
 * to identify exactly where the forward pass breaks down.</p>
 *
 * <p>Qwen3.5-0.8B architecture (28 layers, fullAttentionInterval=4):</p>
 * <ul>
 *   <li>Layers 0,1,2  → linear_attention (GDN/Gated Delta Rule)</li>
 *   <li>Layer 3       → full_attention (first full transformer block)</li>
 *   <li>Layers 4,5,6  → linear_attention (GDN)</li>
 *   <li>Layer 7       → full_attention (second full transformer block)</li>
 *   <li>... and so on every 4th layer</li>
 *   <li>Layer 27      → full_attention (last layer)</li>
 * </ul>
 *
 * <p>Run command:</p>
 * <pre>
 *   cd platform-tests && mvn test -Dtest=TestQwenLayerDiagnostics#testLayerDiagnostics \
 *     2>&1 | tee /tmp/qwen-layer-diag.log
 * </pre>
 */
public class TestQwenLayerDiagnostics {

    // Qwen3.5-0.8B architecture constants (known from GGUF metadata)
    private static final int NUM_LAYERS = 28;
    private static final int FULL_ATTENTION_INTERVAL = 4;
    // Sequence length for prefill
    private static final int SEQ_LEN = 4;
    // KV cache size: prefill length + some decode headroom
    private static final int MAX_KV_LEN = 512;

    /**
     * Run a single prefill forward pass and capture intermediate hidden states
     * at strategic checkpoints to find where the forward pass breaks.
     */
    @Test
    public void testLayerDiagnostics() throws Exception {
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║   QWEN3.5-0.8B LAYER-BY-LAYER DIAGNOSTIC                    ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝");
        System.out.println();

        // ── Step 1: Load model ──────────────────────────────────────────────
        System.out.println("=== STEP 1: MODEL LOAD ===");
        System.out.println("Downloading Qwen3.5-0.8B Q4_K_M...");
        DownloadResult download = LLMModelDownloader.download(LLMModel.QWEN35_0_8B, QuantType.Q4_K_M);
        System.out.println("  Model file: " + download.getModelFile().getAbsolutePath());
        System.out.println("  Size: " + formatBytes(download.getFileSizeBytes()));
        System.out.println("  Cached: " + !download.isDownloadedNow());

        System.out.println("Importing model...");
        long t0 = System.currentTimeMillis();
        SameDiff model = GGMLModelImport.importModel(
                download.getModelFile().getAbsolutePath(),
                ConversionOptions.forInference());
        System.out.println("  Import time: " + (System.currentTimeMillis() - t0) + "ms");
        System.out.println("  Ops: " + model.ops().length);
        System.out.println("  Variables: " + model.variables().size());
        System.out.println("  Inputs: " + model.inputs());
        System.out.println("  Outputs (registered): " + model.outputs());
        System.out.println();

        // Discover actual number of layers from the model's input variables
        // (may differ from constant above if the model has a different config)
        int numLayers = discoverNumLayers(model);
        System.out.println("  Discovered numLayers=" + numLayers + " (from past_key_values placeholders)");
        System.out.println();

        // ── Step 2: Build inputs ────────────────────────────────────────────
        System.out.println("=== STEP 2: BUILD PREFILL INPUTS ===");

        // Input tokens: "Hello, I am" — exact tokens don't matter, we're diagnosing the forward pass
        // Token IDs chosen as reasonable BPE approximations for Qwen's tokenizer
        int[] tokenIds = {9707, 11, 358, 1079};
        System.out.println("  Input token IDs: " + Arrays.toString(tokenIds));
        System.out.println("  Sequence length: " + SEQ_LEN);
        System.out.println("  Max KV cache length: " + MAX_KV_LEN);

        Map<String, INDArray> inputMap = buildPrefillInputs(model, tokenIds, numLayers);

        System.out.println("  Built " + inputMap.size() + " input tensors:");
        for (Map.Entry<String, INDArray> e : inputMap.entrySet()) {
            System.out.println("    " + e.getKey() + " -> shape=" +
                    Arrays.toString(e.getValue().shape()) + " dtype=" + e.getValue().dataType());
        }
        System.out.println();

        // ── Step 3: Define diagnostic checkpoints ───────────────────────────
        System.out.println("=== STEP 3: DIAGNOSTIC CHECKPOINTS ===");

        // Key checkpoint variable names in the SameDiff graph
        // These are named during LLaMAArchitecture.buildGraph()
        List<String> checkpoints = new ArrayList<>();
        checkpoints.add("embedded");       // after token embedding lookup (gather)
        checkpoints.add("post_attn_0");    // after layer 0 (GDN) attention + residual
        checkpoints.add("layer_out_0");    // after layer 0 FFN + residual
        checkpoints.add("post_attn_3");    // after layer 3 (first full attention) + residual
        checkpoints.add("layer_out_3");    // after layer 3 FFN + residual
        checkpoints.add("post_attn_7");    // after layer 7 (second full attention) + residual
        checkpoints.add("layer_out_7");    // after layer 7 FFN + residual
        checkpoints.add("post_attn_11");   // after layer 11 (third full attention) + residual
        checkpoints.add("layer_out_11");   // after layer 11 FFN + residual
        // Add final-layer checkpoints — last full-attn layer depends on numLayers
        // For 28-layer Qwen3.5-0.8B: interval=4, so layer 27 is the last full-attn layer
        // For older 24-layer configs: layer 23 is the last
        if (numLayers > 26) {
            checkpoints.add("post_attn_27");   // after layer 27 (last full attention) + residual
            checkpoints.add("layer_out_27");   // after layer 27 FFN + residual
        } else if (numLayers > 22) {
            checkpoints.add("post_attn_23");
            checkpoints.add("layer_out_23");
        } else {
            // Generic: last layer
            int lastLayer = numLayers - 1;
            checkpoints.add("post_attn_" + lastLayer);
            checkpoints.add("layer_out_" + lastLayer);
        }
        checkpoints.add("lm_logits");      // final output logits

        System.out.println("  Requesting " + checkpoints.size() + " checkpoints:");
        for (String cp : checkpoints) {
            boolean inGraph = model.hasVariable(cp);
            System.out.println("    " + cp + " -> " + (inGraph ? "FOUND in graph" : "*** NOT FOUND ***"));
        }
        System.out.println();

        // Filter to only checkpoints that actually exist in the graph
        List<String> validCheckpoints = new ArrayList<>();
        for (String cp : checkpoints) {
            if (model.hasVariable(cp)) {
                validCheckpoints.add(cp);
            } else {
                System.out.println("  WARNING: Checkpoint '" + cp + "' not found — skipping");
            }
        }

        if (validCheckpoints.isEmpty()) {
            fail("No valid checkpoints found in graph. " +
                    "The model graph may have been renamed or restructured.");
        }

        // ── Step 4: Run forward pass ─────────────────────────────────────────
        System.out.println("=== STEP 4: FORWARD PASS ===");
        System.out.println("  Running decoder.output() with " + validCheckpoints.size() + " outputs...");

        t0 = System.currentTimeMillis();
        Map<String, INDArray> outputs;
        try {
            outputs = model.output(inputMap, validCheckpoints.toArray(new String[0]));
        } catch (Exception e) {
            System.out.println("  FORWARD PASS FAILED: " + e.getClass().getSimpleName() + ": " + e.getMessage());
            e.printStackTrace();
            throw e;
        }
        System.out.println("  Forward pass time: " + (System.currentTimeMillis() - t0) + "ms");
        System.out.println("  Returned " + outputs.size() + " output tensors");
        System.out.println();

        // ── Step 5: Analyze each checkpoint ──────────────────────────────────
        System.out.println("=== STEP 5: CHECKPOINT ANALYSIS ===");
        System.out.println();

        boolean anyNaN = false;
        boolean anyInf = false;
        String firstNaNCheckpoint = null;

        for (String cp : validCheckpoints) {
            // Skip lm_logits here — it gets its own dedicated section in Step 6
            if ("lm_logits".equals(cp)) continue;

            INDArray arr = outputs.get(cp);
            if (arr == null) {
                System.out.println("[" + cp + "] NOT RETURNED by model.output()");
                continue;
            }

            System.out.println("──── Checkpoint: " + cp + " ────");
            printCheckpointStats(cp, arr, SEQ_LEN);

            // Track NaN/Inf progression
            boolean hasNaN = hasNaN(arr);
            boolean hasInf = hasInf(arr);
            if (hasNaN || hasInf) {
                anyNaN = anyNaN || hasNaN;
                anyInf = anyInf || hasInf;
                if (firstNaNCheckpoint == null) {
                    firstNaNCheckpoint = cp;
                    System.out.println("  *** NaN/Inf FIRST DETECTED HERE ***");
                }
            }
            System.out.println();

            arr.close();
        }

        // ── Step 6: Logits-specific analysis ─────────────────────────────────
        System.out.println("=== STEP 6: LOGITS ANALYSIS ===");
        INDArray logitsOutput = outputs.get("lm_logits");
        double peakednessForAssert = Double.NaN;
        if (logitsOutput == null) {
            System.out.println("  lm_logits NOT RETURNED — cannot analyze");
        } else {
            // Also track NaN/Inf in logits
            if (hasNaN(logitsOutput) || hasInf(logitsOutput)) {
                anyNaN = anyNaN || hasNaN(logitsOutput);
                anyInf = anyInf || hasInf(logitsOutput);
                if (firstNaNCheckpoint == null) {
                    firstNaNCheckpoint = "lm_logits";
                }
            }

            analyzeLogits(logitsOutput, SEQ_LEN);

            // Extract peakedness for assertion while logits are still open
            if (logitsOutput.shape().length == 3) {
                long lastPos = logitsOutput.shape()[1] - 1;
                INDArray lastPosLogits = logitsOutput.get(
                        NDArrayIndex.point(0),
                        NDArrayIndex.point(lastPos),
                        NDArrayIndex.all()).dup();
                peakednessForAssert = lastPosLogits.maxNumber().doubleValue()
                        - lastPosLogits.meanNumber().doubleValue();
                lastPosLogits.close();
            }
            logitsOutput.close();
        }
        System.out.println();

        // ── Step 7: Summary ───────────────────────────────────────────────────
        System.out.println("=== STEP 7: DIAGNOSTIC SUMMARY ===");
        if (firstNaNCheckpoint != null) {
            System.out.println("  ROOT CAUSE: NaN/Inf first appeared at checkpoint: " + firstNaNCheckpoint);
            System.out.println("  The forward pass breaks at or before: " + firstNaNCheckpoint);
        } else if (anyNaN || anyInf) {
            System.out.println("  NaN/Inf detected but could not pinpoint layer.");
        } else {
            System.out.println("  No NaN/Inf detected — tensor values are finite throughout.");
        }

        // ── Step 8: Assert logit quality ─────────────────────────────────────
        System.out.println();
        System.out.println("=== STEP 8: LOGIT QUALITY ASSERTION ===");

        if (!Double.isNaN(peakednessForAssert)) {
            System.out.println("  Peakedness (max - mean) = " + String.format("%.4f", peakednessForAssert));
            if (peakednessForAssert < 3.0) {
                System.out.println("  FAIL: Peakedness < 3.0 — logit distribution is too flat (model produces garbage)");
                System.out.println("  A well-calibrated model should have max - mean > 5.0");
            } else {
                System.out.println("  PASS: Peakedness >= 3.0 — logit distribution has signal");
            }

            // Assertion: if peakedness is below threshold, the model is broken
            assertTrue(peakednessForAssert >= 3.0,
                    "Logit distribution is too flat: max - mean = " + String.format("%.4f", peakednessForAssert) +
                    " (expected >= 3.0). Model is producing garbage tokens. " +
                    "Check checkpoint analysis above for the layer where hidden states diverge.");
        } else {
            System.out.println("  Could not compute peakedness — lm_logits unavailable or wrong shape.");
        }

        // Clean up remaining input tensors
        for (INDArray arr : inputMap.values()) {
            if (arr != null) arr.close();
        }

        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║   DIAGNOSTIC COMPLETE                                        ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝");
    }

    // =========================================================================
    // Input construction
    // =========================================================================

    /**
     * Build the complete prefill input map for a single forward pass.
     *
     * <p>Mirrors what GenerationPipeline.generateSimpleWithInGraphKvCache() does for prefill:
     * all KV caches are empty (Nd4j.empty()), GDN states are zero-initialized,
     * and causal mask is a lower-triangular attention bias.</p>
     *
     * @param model      the imported SameDiff model
     * @param tokenIds   token IDs to process
     * @param numLayers  number of transformer layers in the model
     * @return           map from placeholder name to INDArray
     */
    private Map<String, INDArray> buildPrefillInputs(SameDiff model, int[] tokenIds, int numLayers) {
        Map<String, INDArray> inputs = new HashMap<>();
        int seqLen = tokenIds.length;

        // input_ids: [1, seqLen] INT64
        int[][] tokenIds2D = new int[1][seqLen];
        for (int i = 0; i < seqLen; i++) tokenIds2D[0][i] = tokenIds[i];
        INDArray inputIds = Nd4j.createFromArray(tokenIds2D).castTo(DataType.INT64);
        inputs.put("input_ids", inputIds);

        // position_offset: scalar INT64 = 0 (prefill starts at position 0)
        if (model.hasVariable("position_offset")) {
            inputs.put("position_offset", Nd4j.scalar(DataType.INT64, 0L));
        }

        // cache_position: scalar INT64 = 0 (write position for KV cache scatter)
        if (model.hasVariable("cache_position")) {
            inputs.put("cache_position", Nd4j.scalar(DataType.INT64, 0L));
        }

        // _causal_mask: [1, 1, seqLen, maxKvLen]
        // Lower-triangular: position i can attend to positions 0..i, future positions masked
        if (model.hasVariable("_causal_mask")) {
            DataType maskDtype = model.getVariable("_causal_mask").dataType();
            INDArray causalMask = buildCausalMask(seqLen, MAX_KV_LEN, maskDtype);
            inputs.put("_causal_mask", causalMask);
        }

        // Per-layer KV cache and GDN state inputs
        for (int layer = 0; layer < numLayers; layer++) {
            boolean isFullAttention = isFullAttentionLayer(layer);

            if (isFullAttention) {
                // KV cache inputs: empty signals attention to skip in-place scatter during prefill
                String keyName = "past_key_values." + layer + ".key";
                String valName = "past_key_values." + layer + ".value";
                if (model.hasVariable(keyName)) {
                    DataType kvDtype = model.getVariable(keyName).dataType();
                    inputs.put(keyName, Nd4j.empty(kvDtype));
                }
                if (model.hasVariable(valName)) {
                    DataType kvDtype = model.getVariable(valName).dataType();
                    inputs.put(valName, Nd4j.empty(kvDtype));
                }
            } else {
                // GDN (linear_attention) state inputs: zero-initialized for prefill
                String gdnName = "past_gdn_state." + layer;
                String convName = "past_conv_state." + layer;
                if (model.hasVariable(gdnName)) {
                    long[] gdnShape = computeGdnStateShape(model, layer);
                    DataType gdnDtype = model.getVariable(gdnName).dataType();
                    inputs.put(gdnName, Nd4j.zeros(gdnDtype, gdnShape));
                }
                if (model.hasVariable(convName)) {
                    long[] convShape = computeConvStateShape(model, layer);
                    DataType convDtype = model.getVariable(convName).dataType();
                    inputs.put(convName, Nd4j.zeros(convDtype, convShape));
                }
            }
        }

        return inputs;
    }

    /**
     * Determine if a layer is full_attention (every FULL_ATTENTION_INTERVAL-th layer, 0-indexed).
     * Qwen3.5: interval=4, so layers 3, 7, 11, 15, 19, 23 are full_attention.
     */
    private boolean isFullAttentionLayer(int layerIdx) {
        return (layerIdx + 1) % FULL_ATTENTION_INTERVAL == 0;
    }

    /**
     * Build lower-triangular causal mask for prefill.
     * Shape: [1, 1, prefillLen, maxKvLen]
     * Position i can attend to 0..i; future positions and padding get large negative value.
     */
    private INDArray buildCausalMask(int prefillLen, int maxKvLen, DataType dtype) {
        // Use dtype-safe mask value: avoid -inf overflow in FP16
        float maskVal = (dtype == DataType.HALF || dtype == DataType.FLOAT16) ? -65504.0f : -1e9f;
        float[] data = new float[prefillLen * maxKvLen];

        for (int q = 0; q < prefillLen; q++) {
            int rowOffset = q * maxKvLen;
            // Mask positions after the current query position AND all padding
            for (int k = q + 1; k < maxKvLen; k++) {
                data[rowOffset + k] = maskVal;
            }
        }

        INDArray mask = Nd4j.create(data, new long[]{1, 1, prefillLen, maxKvLen}, 'c');
        if (dtype != DataType.FLOAT) {
            INDArray cast = mask.castTo(dtype);
            mask.close();
            return cast;
        }
        return mask;
    }

    /**
     * Compute GDN state shape for a GDN layer.
     * Shape: [1, numGdnHeads, headDimKV, headDimKV]
     * Derived from ssm_a weight (shape [H]) and qkv weight (shape [qkvDim, ...]).
     */
    private long[] computeGdnStateShape(SameDiff model, int layerIdx) {
        try {
            String ssmAName = "model.layers." + layerIdx + ".gdn.a";
            String qkvName  = "model.layers." + layerIdx + ".gdn.qkv.weight";
            if (model.hasVariable(ssmAName) && model.hasVariable(qkvName)) {
                long numGdnHeads = model.getVariable(ssmAName).getArr().shape()[0];
                long qkvDim      = model.getVariable(qkvName).getArr().shape()[0];
                long headDimKV   = qkvDim / (3 * numGdnHeads);
                return new long[]{1, numGdnHeads, headDimKV, headDimKV};
            }
        } catch (Exception e) {
            System.out.println("  WARNING: could not compute GDN state shape for layer " + layerIdx + ": " + e.getMessage());
        }
        // Fallback: use placeholder shape from the variable if defined
        long[] placeholderShape = model.getVariable("past_gdn_state." + layerIdx).getShape();
        if (placeholderShape != null && placeholderShape.length == 4 && placeholderShape[0] > 0) {
            return placeholderShape;
        }
        // Last resort: use known Qwen3.5-0.8B defaults
        // [1, 16, 128, 128] — 16 GDN heads, 128-dim K/V
        return new long[]{1, 16, 128, 128};
    }

    /**
     * Compute causal conv state shape for a GDN layer.
     * Shape: [1, convDim, kernelSize - 1]
     * Derived from conv weight (shape [D, K, ...]).
     */
    private long[] computeConvStateShape(SameDiff model, int layerIdx) {
        try {
            String convWeightName = "model.layers." + layerIdx + ".gdn.conv.weight";
            if (model.hasVariable(convWeightName)) {
                long[] convShape  = model.getVariable(convWeightName).getArr().shape();
                long convDim      = convShape[0];   // D
                long kernelSize   = convShape[1];   // K
                return new long[]{1, convDim, kernelSize - 1};
            }
        } catch (Exception e) {
            System.out.println("  WARNING: could not compute conv state shape for layer " + layerIdx + ": " + e.getMessage());
        }
        // Fallback: use known Qwen3.5-0.8B defaults
        // [1, 6144, 3] — 6144-dim conv, kernel_size=4 → state=3
        return new long[]{1, 6144, 3};
    }

    // =========================================================================
    // Checkpoint statistics
    // =========================================================================

    /**
     * Print detailed statistics for a checkpoint tensor.
     * For hidden states [1, seqLen, hiddenDim], prints last-position stats.
     * For logits [1, seqLen, vocabSize], delegates to analyzeLogits.
     */
    private void printCheckpointStats(String name, INDArray arr, int seqLen) {
        long[] shape = arr.shape();
        DataType dtype = arr.dataType();

        System.out.println("  Shape: " + Arrays.toString(shape));
        System.out.println("  Dtype: " + dtype);

        // On CPU, data is already on host. On CUDA, dup() forces sync.

        // Compute stats over the full array
        double minVal   = arr.minNumber().doubleValue();
        double maxVal   = arr.maxNumber().doubleValue();
        double meanVal  = arr.meanNumber().doubleValue();
        double stdVal   = computeStd(arr);
        boolean hasNaN  = hasNaN(arr);
        boolean hasInf  = hasInf(arr);

        System.out.printf("  Stats: min=%.4f  max=%.4f  mean=%.6f  std=%.4f%n",
                minVal, maxVal, meanVal, stdVal);
        System.out.println("  NaN: " + hasNaN + "  Inf: " + hasInf);

        // Print first 10 values of the last-position hidden state
        // For shape [1, seqLen, hiddenDim] or [1, seqLen, vocabSize]
        if (shape.length == 3 && shape[0] == 1 && shape[1] >= seqLen) {
            long lastPos = shape[1] - 1;
            INDArray lastPosSlice = arr.get(NDArrayIndex.point(0), NDArrayIndex.point(lastPos), NDArrayIndex.all());
            long printLen = Math.min(10, lastPosSlice.length());

            StringBuilder sb = new StringBuilder("  Last-pos first " + printLen + " values: [");
            for (int i = 0; i < printLen; i++) {
                if (i > 0) sb.append(", ");
                sb.append(String.format("%.4f", lastPosSlice.getFloat(i)));
            }
            sb.append("]");
            System.out.println(sb.toString());
        } else if (shape.length == 2 && shape[0] >= seqLen) {
            // 2D: [seqLen, hiddenDim] — print last row
            long lastPos = shape[0] - 1;
            INDArray lastRow = arr.get(NDArrayIndex.point(lastPos), NDArrayIndex.all());
            long printLen = Math.min(10, lastRow.length());

            StringBuilder sb = new StringBuilder("  Last-pos first " + printLen + " values: [");
            for (int i = 0; i < printLen; i++) {
                if (i > 0) sb.append(", ");
                sb.append(String.format("%.4f", lastRow.getFloat(i)));
            }
            sb.append("]");
            System.out.println(sb.toString());
        }
    }

    /**
     * Detailed logit analysis: top-5 tokens, peakedness, distribution shape.
     */
    private void analyzeLogits(INDArray logits, int seqLen) {
        long[] shape = logits.shape();
        System.out.println("  Logits shape: " + Arrays.toString(shape));
        System.out.println("  Logits dtype: " + logits.dataType());

        if (shape.length != 3) {
            System.out.println("  WARNING: unexpected logits rank " + shape.length + " (expected 3)");
            return;
        }

        // On CPU, data is already on host.

        long seqDim   = shape[1];
        long vocabSize = shape[2];
        long lastPos  = seqDim - 1;

        // Extract last-position logits [vocabSize]
        INDArray lastPosLogits = logits.get(
                NDArrayIndex.point(0),
                NDArrayIndex.point(lastPos),
                NDArrayIndex.all()).dup();

        double minLogit  = lastPosLogits.minNumber().doubleValue();
        double maxLogit  = lastPosLogits.maxNumber().doubleValue();
        double meanLogit = lastPosLogits.meanNumber().doubleValue();
        double stdLogit  = computeStd(lastPosLogits);
        double peakedness = maxLogit - meanLogit;

        System.out.println("  Last-position logits [vocabSize=" + vocabSize + "]:");
        System.out.printf("    min=%.4f  max=%.4f  mean=%.4f  std=%.4f%n",
                minLogit, maxLogit, meanLogit, stdLogit);
        System.out.printf("    peakedness (max - mean) = %.4f%n", peakedness);

        // Interpretation
        System.out.println();
        if (peakedness < 1.0) {
            System.out.println("  *** DIAGNOSIS: Distribution is nearly UNIFORM (peakedness < 1.0)");
            System.out.println("  *** This indicates complete hidden state corruption.");
        } else if (peakedness < 3.0) {
            System.out.println("  *** DIAGNOSIS: Distribution is VERY FLAT (1.0 <= peakedness < 3.0)");
            System.out.println("  *** Model produces near-random output. Significant corruption present.");
        } else if (peakedness < 5.0) {
            System.out.println("  *** DIAGNOSIS: Distribution is SOMEWHAT FLAT (3.0 <= peakedness < 5.0)");
            System.out.println("  *** Model has partial signal but may not generate coherent text.");
        } else {
            System.out.println("  DIAGNOSIS: Distribution appears REASONABLE (peakedness >= 5.0)");
            System.out.println("  The flat logit issue may be in token sampling, not forward pass.");
        }
        System.out.println();

        // Top-5 tokens
        System.out.println("  Top-5 tokens (by logit value):");
        float[] logitArr = lastPosLogits.data().asFloat();
        int[] top5 = topK(logitArr, 5);
        for (int rank = 0; rank < top5.length; rank++) {
            int tokenId = top5[rank];
            System.out.printf("    rank %d: token_id=%d  logit=%.4f%n",
                    rank + 1, tokenId, logitArr[tokenId]);
        }
        System.out.println();

        // Argmax logit vs mean
        int argmaxToken = top5[0];
        System.out.printf("  Argmax token: id=%d  logit=%.4f%n", argmaxToken, logitArr[argmaxToken]);
        System.out.printf("  Mean logit:   %.4f%n", meanLogit);
        System.out.printf("  Argmax - mean = %.4f (peakedness)%n", peakedness);

        // Also compute "softmax temperature" proxy: how much of probability mass is on top token
        // Approximate softmax for top token vs total using sum of exp(logit_i - max)
        double sumExp = 0.0;
        float maxF = (float) maxLogit;
        for (float v : logitArr) {
            sumExp += Math.exp(v - maxF);
        }
        double topProb = 1.0 / sumExp;
        System.out.printf("  Approx top-1 probability (greedy softmax) = %.6f%n", topProb);
        System.out.printf("  Expected for random: ~%.6f  (1/vocabSize = 1/%d)%n", 1.0 / vocabSize, vocabSize);

        if (topProb < 2.0 / vocabSize) {
            System.out.println("  *** WARNING: Top-1 probability is near-random — model output is garbage");
        }

        lastPosLogits.close();
    }

    // =========================================================================
    // Utility helpers
    // =========================================================================

    /** Discover the number of transformer layers from KV cache or GDN placeholders. */
    private int discoverNumLayers(SameDiff model) {
        int maxLayer = -1;
        for (String inputName : model.inputs()) {
            if (inputName.startsWith("past_key_values.") || inputName.startsWith("past_gdn_state.")) {
                // Extract layer index
                String[] parts = inputName.split("\\.");
                if (parts.length >= 2) {
                    try {
                        int layerIdx = Integer.parseInt(parts[1]);
                        maxLayer = Math.max(maxLayer, layerIdx);
                    } catch (NumberFormatException ignored) {
                    }
                }
            }
        }
        return maxLayer >= 0 ? maxLayer + 1 : NUM_LAYERS;
    }

    /** Compute standard deviation of an INDArray. */
    private double computeStd(INDArray arr) {
        try {
            double mean = arr.meanNumber().doubleValue();
            INDArray diff = arr.sub(mean);
            INDArray sq = diff.mul(diff);
            double variance = sq.meanNumber().doubleValue();
            diff.close();
            sq.close();
            return Math.sqrt(variance);
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    /** Check if any element is NaN. */
    private boolean hasNaN(INDArray arr) {
        try {
            return Double.isNaN(arr.sumNumber().doubleValue());
        } catch (Exception e) {
            return true;
        }
    }

    /** Check if any element is +/-Inf. */
    private boolean hasInf(INDArray arr) {
        try {
            double max = arr.maxNumber().doubleValue();
            double min = arr.minNumber().doubleValue();
            return Double.isInfinite(max) || Double.isInfinite(min);
        } catch (Exception e) {
            return false;
        }
    }

    /** Return indices of the top-k values in a float array. */
    private int[] topK(float[] arr, int k) {
        k = Math.min(k, arr.length);
        // Simple selection for small k
        Integer[] indices = new Integer[arr.length];
        for (int i = 0; i < arr.length; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Float.compare(arr[b], arr[a]));
        int[] top = new int[k];
        for (int i = 0; i < k; i++) top[i] = indices[i];
        return top;
    }

    /** Format bytes for display. */
    private static String formatBytes(long bytes) {
        if (bytes < 1024L * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024L * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }
}
