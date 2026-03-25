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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.DownloadResult;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.LLMModel;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.QuantType;
import org.nd4j.autodiff.samediff.SDVariable;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test for text-only GenerationPipeline using TextGenerator with Qwen3.5 GGUF models.
 *
 * <p>This validates the text generation pipeline end-to-end:
 * download GGUF model, import to SameDiff, tokenize, generate text, validate output.</p>
 *
 * <p>System properties for control:</p>
 * <ul>
 *   <li>{@code -Dtext.model.size=0.8B} - model size (default: 0.8B)</li>
 *   <li>{@code -Dtext.quant=Q4_K_M} - quantization type (default: Q4_K_M)</li>
 *   <li>{@code -Dtext.max.tokens=20} - max generation tokens (default: 20)</li>
 *   <li>{@code -Dtext.prompt="..."} - custom prompt (default: "What is the capital of France?")</li>
 * </ul>
 *
 * <p>Run with:</p>
 * <pre>
 * cd platform-tests && mvn test \
 *   -Dtest=TestTextGenerationPipeline \
 *   -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
public class TestTextGenerationPipeline {

    private static final String DEFAULT_PROMPT = "What is the capital of France?";

    @BeforeAll
    public static void setup() {
        if (System.getProperty("nd4j.optimizer.enabled") == null) {
            System.setProperty("nd4j.optimizer.enabled", "true");
        }
    }

    @Test
    @DisplayName("Text GenerationPipeline: TextGenerator with Qwen3.5 GGUF")
    public void testTextGenerationWithTextGenerator() throws Exception {
        String sizeLabel = System.getProperty("text.model.size", "0.8B");
        String quantStr = System.getProperty("text.quant", "Q4_K_M");
        int maxTokens = Integer.parseInt(System.getProperty("text.max.tokens", "20"));
        String prompt = System.getProperty("text.prompt", DEFAULT_PROMPT);

        LLMModel llmModel = LLMModel.fromSizeLabel(sizeLabel);
        QuantType quantType = QuantType.valueOf(quantStr);

        // --- 1. Download model ---
        log.info("Downloading Qwen3.5 {} {} ...", sizeLabel, quantType);
        long t0 = System.currentTimeMillis();
        DownloadResult downloadResult = LLMModelDownloader.download(llmModel, quantType);
        long downloadMs = System.currentTimeMillis() - t0;
        log.info("Download: {}ms (cached={})", downloadMs, !downloadResult.isDownloadedNow());

        assertTrue(downloadResult.getModelFile().exists(), "Model file must exist");

        // --- 1b. Inspect GGUF metadata ---
        org.nd4j.ggml.format.GGMLMetadata meta = GGMLModelImport.inspectModel(downloadResult.getModelFile());
        System.out.println("[META-DIAG] architecture=" + meta.getArchitecture());
        System.out.println("[META-DIAG] numLayers=" + meta.getNumLayers());
        System.out.println("[META-DIAG] hiddenSize=" + meta.getHiddenSize());
        System.out.println("[META-DIAG] numAttentionHeads=" + meta.getNumAttentionHeads());
        System.out.println("[META-DIAG] numKVHeads=" + meta.getNumKVHeads());
        System.out.println("[META-DIAG] attentionKeyLength=" + meta.getAttentionKeyLength());
        System.out.println("[META-DIAG] fullAttentionInterval=" + meta.getFullAttentionInterval());
        System.out.println("[META-DIAG] layerTypes=" + meta.getLayerTypes());
        System.out.println("[META-DIAG] ropeFreqBase=" + meta.getRopeFreqBase());
        System.out.println("[META-DIAG] ropeDimensionCount=" + meta.getRopeDimensionCount());
        System.out.println("[META-DIAG] vocabSize=" + meta.getVocabSize());
        System.out.println("[META-DIAG] contextLength=" + meta.getContextLength());
        // Show some raw metadata keys to check for architecture prefix issues
        java.util.Map<String, Object> rawMeta = meta.getRawMetadata();
        for (String key : rawMeta.keySet()) {
            if (key.contains("full_attention") || key.contains("layer_type") || key.contains("architecture")
                || key.contains("block_count") || key.contains("ssm") || key.contains("key_length")) {
                System.out.println("[RAW-META] " + key + "=" + rawMeta.get(key));
            }
        }
        // Show tensor names for layer 0 and layer 3 (full attention)
        for (org.nd4j.ggml.format.GGMLTensorInfo ti : meta.getTensors()) {
            String tn = ti.getName();
            if (tn.startsWith("blk.0.") || tn.startsWith("blk.3.") || tn.equals("token_embd.weight") || tn.equals("output.weight")) {
                System.out.println("[TENSOR] " + tn + " type=" + ti.getDataType() + " shape=" + java.util.Arrays.toString(ti.getShape()));
            }
        }

        // --- 2. Import GGUF -> SameDiff ---
        log.info("Importing GGUF model...");
        t0 = System.currentTimeMillis();
        SameDiff model = GGMLModelImport.importModel(downloadResult.getModelFile().getAbsolutePath(),
                org.nd4j.ggml.convert.ConversionOptions.builder()
                        .quantizationMode(org.nd4j.ggml.convert.ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT32)
                        .targetDataType(DataType.FLOAT)
                        .build());
        long importMs = System.currentTimeMillis() - t0;
        int opCount = model.ops() != null ? model.ops().length : 0;
        log.info("Import: {}ms, {} ops", importMs, opCount);

        assertTrue(opCount > 0, "Model should have ops after import");

        // --- 2b. Diagnostic: check weight values ---
        for (SDVariable sdVar : model.variables()) {
            String varName = sdVar.name();
            if (varName.contains("embed") || varName.contains("layers.0.")) {
                org.nd4j.linalg.api.ndarray.INDArray arr = model.getVariable(varName).getArr();
                if (arr != null) {
                    // Use System.out so surefire captures it
                    System.out.println("[WEIGHT-DIAG] " + varName
                        + " dtype=" + arr.dataType()
                        + " shape=" + java.util.Arrays.toString(arr.shape())
                        + " isView=" + arr.isView()
                        + " length=" + arr.length()
                        + " amean=" + arr.castTo(DataType.FLOAT).ameanNumber().floatValue()
                        + " min=" + arr.castTo(DataType.FLOAT).minNumber().floatValue()
                        + " max=" + arr.castTo(DataType.FLOAT).maxNumber().floatValue()
                    );
                }
            }
        }

        // --- 3. Load tokenizer ---
        t0 = System.currentTimeMillis();
        String tokenizerUrl = "https://huggingface.co/Qwen/Qwen3.5-" + sizeLabel + "/resolve/main/tokenizer.json";
        File tokenizerFile = LLMModelDownloader.downloadCustom(tokenizerUrl, "qwen35-" + sizeLabel + "-tokenizer.json");
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile.getAbsolutePath());
        long tokMs = System.currentTimeMillis() - t0;
        log.info("Tokenizer: {}ms, vocab={}", tokMs, tokenizer.getVocabSize());

        assertTrue(tokenizer.getVocabSize() > 0, "Tokenizer must have vocabulary");

        // --- 4. Configure model ---
        model.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        model.setDspAutoCompileEnabled(false);
        model.setDspNativeAutoCompileEnabled(false);

        // Debug+verbose disabled for clean generation test
        // Nd4j.getEnvironment().setDebug(true);
        // Nd4j.getEnvironment().setVerbose(true);

        // --- 4b. Diagnostic: trace intermediate values through first forward pass ---
        {
            // Simple 3-token input: just BOS + "Hello"
            org.eclipse.deeplearning4j.llm.tokenizer.Encoding enc = tokenizer.encode("Hello", true);
            int[] ids = enc.getIds();
            System.out.println("[TRACE] Input token IDs: " + java.util.Arrays.toString(ids));
            org.nd4j.linalg.api.ndarray.INDArray testInput = Nd4j.createFromArray(ids).castTo(DataType.INT64).reshape(1, ids.length);
            System.out.println("[TRACE] input_ids shape=" + java.util.Arrays.toString(testInput.shape()) + " dtype=" + testInput.dataType());

            // Find interesting intermediate variable names
            for (SDVariable v : model.variables()) {
                String n = v.name();
                if (n.equals("embedded") || n.equals("logits")
                    || n.startsWith("post_attn_0") || n.startsWith("layer_out_0")
                    || n.startsWith("gdn_conv_0") || n.startsWith("gdn_qkv_0")
                    || n.startsWith("post_attn_3") || n.startsWith("layer_out_3")
                    || n.startsWith("attn_out_0") || n.startsWith("attn_out_3")
                    || n.startsWith("q_rope_") || n.startsWith("gdn_flat_0")
                    || n.equals("model.norm")) {
                    System.out.println("[TRACE] Found var: " + n);
                }
            }

            // Run model and get logits + key intermediate outputs
            java.util.List<String> evalVars = new java.util.ArrayList<>();
            evalVars.add("logits");
            // Trace ALL layer outputs to find where corruption starts
            java.util.List<String> traceNameList = new java.util.ArrayList<>();
            traceNameList.add("embedded");
            // GDN layer 0 internals
            traceNameList.addAll(java.util.Arrays.asList(
                "gdn_qkv_0", "gdn_conv_0",
                "gdn_q_0", "gdn_k_0", "gdn_v_0",
                "gdn_q_l2norm_0", "gdn_k_l2norm_0",
                "gdn_beta_0", "gdn_gate_decay_0",
                "gdn_out_0", "gdn_gated_0",
                "gdn_flat_0", "gdn_proj_0"
            ));
            // ALL layer outputs (post_attn and layer_out for each layer)
            for (int li = 0; li < 24; li++) {
                traceNameList.add("post_attn_" + li);
                traceNameList.add("layer_out_" + li);
                // FFN outputs
                traceNameList.add("gate_" + li);
                traceNameList.add("down_" + li);
                // GDN-specific for all GDN layers
                traceNameList.add("gdn_out_" + li);
                traceNameList.add("gdn_proj_" + li);
                // Full attention outputs for attn layers
                traceNameList.add("attn_out_" + li);
                traceNameList.add("attn_proj_" + li);
            }
            String[] traceNames = traceNameList.toArray(new String[0]);
            for (SDVariable v : model.variables()) {
                String n = v.name();
                for (String tn : traceNames) {
                    if (n.equals(tn)) {
                        evalVars.add(n);
                        break;
                    }
                }
            }

            java.util.Map<String, org.nd4j.linalg.api.ndarray.INDArray> placeholders = new java.util.HashMap<>();
            placeholders.put("input_ids", testInput);
            java.util.Map<String, org.nd4j.linalg.api.ndarray.INDArray> outputs =
                model.output(placeholders, evalVars.toArray(new String[0]));

            for (java.util.Map.Entry<String, org.nd4j.linalg.api.ndarray.INDArray> entry : outputs.entrySet()) {
                org.nd4j.linalg.api.ndarray.INDArray arr = entry.getValue();
                org.nd4j.linalg.api.ndarray.INDArray f32 = arr.castTo(DataType.FLOAT);
                System.out.println("[TRACE-OUT] " + entry.getKey()
                    + " shape=" + java.util.Arrays.toString(arr.shape())
                    + " dtype=" + arr.dataType()
                    + " amean=" + f32.ameanNumber().floatValue()
                    + " min=" + f32.minNumber().floatValue()
                    + " max=" + f32.maxNumber().floatValue()
                    + " std=" + f32.stdNumber().floatValue());
            }

            // Check logits distribution — top 10 tokens
            org.nd4j.linalg.api.ndarray.INDArray logitsArr = outputs.get("logits");
            if (logitsArr != null && logitsArr.rank() == 3) {
                // Get last position logits
                org.nd4j.linalg.api.ndarray.INDArray lastLogits = logitsArr.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(logitsArr.size(1) - 1),
                    org.nd4j.linalg.indexing.NDArrayIndex.all()).dup().castTo(DataType.FLOAT);
                // Top 10 tokens
                for (int rank = 0; rank < 10; rank++) {
                    org.nd4j.linalg.api.ndarray.INDArray topIdx = Nd4j.argMax(lastLogits.reshape(1, -1), 1);
                    int topToken = topIdx.getInt(0);
                    float topLogit = lastLogits.getFloat(topToken);
                    String decoded = tokenizer.decode(new int[]{topToken}, true);
                    System.out.println("[TOP-" + rank + "] token=" + topToken + " logit=" + topLogit
                        + " decoded='" + decoded + "'");
                    // Zero out this position for next iteration
                    lastLogits.putScalar(topToken, Float.NEGATIVE_INFINITY);
                }
            }

            // --- MANUAL VERIFICATION: Verify QKV projection of layer 0 ---
            System.out.println("\n=== MANUAL VERIFICATION: Layer 0 QKV projection ===");
            {
                org.nd4j.linalg.api.ndarray.INDArray embOut = outputs.get("embedded").dup().castTo(DataType.FLOAT);
                org.nd4j.linalg.api.ndarray.INDArray attnNormW = model.getVariable("model.layers.0.input_layernorm.weight").getArr().dup().castTo(DataType.FLOAT);
                org.nd4j.linalg.api.ndarray.INDArray qkvW = model.getVariable("model.layers.0.gdn.qkv.weight").getArr().dup().castTo(DataType.FLOAT);

                System.out.println("[MANUAL] embedded shape=" + java.util.Arrays.toString(embOut.shape()) + " first 5: " + embOut.get(org.nd4j.linalg.indexing.NDArrayIndex.point(0), org.nd4j.linalg.indexing.NDArrayIndex.point(0), org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 5)));
                System.out.println("[MANUAL] attnNormW shape=" + java.util.Arrays.toString(attnNormW.shape()) + " first 5: " + attnNormW.get(org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 5)));
                System.out.println("[MANUAL] qkvW shape=" + java.util.Arrays.toString(qkvW.shape()) + " amean=" + qkvW.ameanNumber().floatValue());

                // Manual RMSNorm: x / sqrt(mean(x^2) + eps) * gamma
                org.nd4j.linalg.api.ndarray.INDArray x = embOut.reshape(1024);
                org.nd4j.linalg.api.ndarray.INDArray sq = x.mul(x);
                float meanSqr = sq.meanNumber().floatValue();
                float rmsVal = (float) Math.sqrt(meanSqr + 1e-6);
                org.nd4j.linalg.api.ndarray.INDArray normed = x.div(rmsVal).mul(attnNormW);
                System.out.println("[MANUAL] RMS normed first 5: " + normed.get(org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 5))
                    + " amean=" + normed.ameanNumber().floatValue());

                // Manual QKV projection: normed @ W_qkv^T
                // qkvW shape [6144, 1024], so W^T shape [1024, 6144]
                org.nd4j.linalg.api.ndarray.INDArray manualQkv = normed.reshape(1, 1024).mmul(qkvW.transpose());
                manualQkv = manualQkv.reshape(6144);
                System.out.println("[MANUAL] qkv first 5: " + manualQkv.get(org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 5))
                    + " amean=" + manualQkv.ameanNumber().floatValue()
                    + " min=" + manualQkv.minNumber().floatValue()
                    + " max=" + manualQkv.maxNumber().floatValue());

                // Compare with model's gdn_qkv_0
                org.nd4j.linalg.api.ndarray.INDArray modelQkv = outputs.get("gdn_qkv_0");
                if (modelQkv != null) {
                    org.nd4j.linalg.api.ndarray.INDArray modelFlat = modelQkv.dup().castTo(DataType.FLOAT).reshape(6144);
                    System.out.println("[MANUAL] model gdn_qkv_0 first 5: " + modelFlat.get(org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 5)));
                    org.nd4j.linalg.api.ndarray.INDArray diff = manualQkv.sub(modelFlat);
                    System.out.println("[MANUAL] diff amean=" + diff.ameanNumber().floatValue()
                        + " max_abs=" + diff.castTo(DataType.FLOAT).ameanNumber().floatValue());
                    // Check if they're close
                    float maxDiff = Nd4j.math().abs(diff).maxNumber().floatValue();
                    System.out.println("[MANUAL] max_abs_diff=" + maxDiff + " (should be < 0.01 for correct computation)");
                }

                // Manual CausalConv1d for L=1: output = SiLU(weight[:, K-1] * input[:])
                org.nd4j.linalg.api.ndarray.INDArray convW = model.getVariable("model.layers.0.gdn.conv.weight").getArr().dup().castTo(DataType.FLOAT);
                System.out.println("[MANUAL] conv weight shape=" + java.util.Arrays.toString(convW.shape())
                    + " col 0 amean=" + convW.getColumn(0).ameanNumber().floatValue()
                    + " col 3 amean=" + convW.getColumn(3).ameanNumber().floatValue());

                // For L=1, only last kernel position (col 3 for kernel_size=4) contributes
                org.nd4j.linalg.api.ndarray.INDArray lastKernelCol = convW.getColumn(3).dup().reshape(6144);
                org.nd4j.linalg.api.ndarray.INDArray convInput = manualQkv.dup(); // use manual QKV output
                org.nd4j.linalg.api.ndarray.INDArray preActivation = lastKernelCol.mul(convInput);
                // SiLU(x) = x * sigmoid(x)
                org.nd4j.linalg.api.ndarray.INDArray sigmoidVal = Nd4j.nn().sigmoid(preActivation);
                org.nd4j.linalg.api.ndarray.INDArray manualConv = preActivation.mul(sigmoidVal);
                System.out.println("[MANUAL] conv output (col 3) first 5: " + manualConv.get(org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 5))
                    + " amean=" + manualConv.ameanNumber().floatValue());

                // Compare with model's gdn_conv_0
                org.nd4j.linalg.api.ndarray.INDArray modelConv = outputs.get("gdn_conv_0");
                if (modelConv != null) {
                    org.nd4j.linalg.api.ndarray.INDArray modelConvFlat = modelConv.dup().castTo(DataType.FLOAT).reshape(6144);
                    System.out.println("[MANUAL] model gdn_conv_0 first 5: " + modelConvFlat.get(org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 5)));
                    float convMaxDiff = Nd4j.math().abs(manualConv.sub(modelConvFlat)).maxNumber().floatValue();
                    System.out.println("[MANUAL] conv max_abs_diff (col 3)=" + convMaxDiff);

                    // Also try with col 0 to see if kernel orientation is reversed
                    org.nd4j.linalg.api.ndarray.INDArray firstKernelCol = convW.getColumn(0).dup().reshape(6144);
                    org.nd4j.linalg.api.ndarray.INDArray preAct0 = firstKernelCol.mul(convInput);
                    org.nd4j.linalg.api.ndarray.INDArray sig0 = Nd4j.nn().sigmoid(preAct0);
                    org.nd4j.linalg.api.ndarray.INDArray manualConv0 = preAct0.mul(sig0);
                    float convMaxDiff0 = Nd4j.math().abs(manualConv0.sub(modelConvFlat)).maxNumber().floatValue();
                    System.out.println("[MANUAL] conv max_abs_diff (col 0)=" + convMaxDiff0);
                }
            }

            // --- BYPASS TEST: embedding -> final_norm -> lm_head (no transformer layers) ---
            System.out.println("\n=== BYPASS TEST: embedding -> norm -> lm_head (no layers) ===");
            {
                org.nd4j.linalg.api.ndarray.INDArray embedWeight = model.getVariable("model.embed_tokens.weight").getArr();
                org.nd4j.linalg.api.ndarray.INDArray lmHeadWeight = model.getVariable("lm_head.weight").getArr();
                org.nd4j.linalg.api.ndarray.INDArray normWeight = null;
                // Find the final norm weight
                for (SDVariable sv : model.variables()) {
                    if (sv.name().equals("model.norm.weight")) {
                        normWeight = model.getVariable(sv.name()).getArr();
                        break;
                    }
                }

                System.out.println("[BYPASS] embed shape=" + java.util.Arrays.toString(embedWeight.shape()));
                System.out.println("[BYPASS] lm_head shape=" + java.util.Arrays.toString(lmHeadWeight.shape()));
                System.out.println("[BYPASS] normWeight=" + (normWeight != null ? java.util.Arrays.toString(normWeight.shape()) : "NULL"));
                System.out.println("[BYPASS] embed == lm_head? " + (embedWeight == lmHeadWeight));

                // Get embedding for token 9419 ("Hello")
                int tokenId = ids[ids.length - 1]; // Last token (Hello without BOS)
                org.nd4j.linalg.api.ndarray.INDArray emb = embedWeight.getRow(tokenId).dup().castTo(DataType.FLOAT);
                System.out.println("[BYPASS] token " + tokenId + " emb first 10: " + emb.get(org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 10)));

                // Apply RMSNorm manually: x / sqrt(mean(x^2) + eps) * gamma
                org.nd4j.linalg.api.ndarray.INDArray embF = emb.castTo(DataType.FLOAT);
                org.nd4j.linalg.api.ndarray.INDArray squared = embF.mul(embF);
                float meanSq = squared.meanNumber().floatValue();
                float rms = (float) Math.sqrt(meanSq + 1e-6);
                org.nd4j.linalg.api.ndarray.INDArray normalized = embF.div(rms);
                if (normWeight != null) {
                    normalized = normalized.mul(normWeight.castTo(DataType.FLOAT));
                }
                System.out.println("[BYPASS] after norm first 10: " + normalized.get(org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 10)));

                // Compute logits: normalized @ lm_head.T
                org.nd4j.linalg.api.ndarray.INDArray lmF = lmHeadWeight.castTo(DataType.FLOAT);
                // lm_head shape is [vocab, hidden] after reverseShape
                // logits = normalized[1, hidden] @ lm_head.T[hidden, vocab]
                org.nd4j.linalg.api.ndarray.INDArray bypassLogits = normalized.reshape(1, -1).mmul(lmF.transpose());
                bypassLogits = bypassLogits.reshape(-1);
                System.out.println("[BYPASS] logits shape=" + java.util.Arrays.toString(bypassLogits.shape())
                    + " amean=" + bypassLogits.ameanNumber().floatValue()
                    + " min=" + bypassLogits.minNumber().floatValue()
                    + " max=" + bypassLogits.maxNumber().floatValue());

                // Top 10 bypass tokens
                for (int rank = 0; rank < 10; rank++) {
                    org.nd4j.linalg.api.ndarray.INDArray topIdx = Nd4j.argMax(bypassLogits.reshape(1, -1), 1);
                    int topToken = topIdx.getInt(0);
                    float topLogit = bypassLogits.getFloat(topToken);
                    String decoded = tokenizer.decode(new int[]{topToken}, true);
                    System.out.println("[BYPASS-TOP-" + rank + "] token=" + topToken + " logit=" + topLogit
                        + " decoded='" + decoded + "'");
                    bypassLogits.putScalar(topToken, Float.NEGATIVE_INFINITY);
                }
            }

            testInput.close();
        }

        // --- 5. Generate with TextGenerator ---
        TextGenerator generator = TextGenerator.builder()
                .model(model)
                .tokenizer(tokenizer)
                .config(SamplingConfig.greedy())
                .build();

        log.info("Generating with prompt: '{}'", prompt);
        t0 = System.currentTimeMillis();
        GenerationResult result = generator.generateWithMetrics(prompt, maxTokens);
        long genMs = System.currentTimeMillis() - t0;

        // --- 6. Validate ---
        assertNotNull(result, "GenerationResult must not be null");
        assertNotNull(result.getText(), "Generated text must not be null");
        assertTrue(result.getGeneratedTokenCount() > 0,
                "Should generate at least 1 token, got: " + result.getGeneratedTokenCount());
        assertNotNull(result.getTokenIds(), "Token IDs must not be null");
        assertEquals(result.getGeneratedTokenCount(), result.getTokenIds().length,
                "Token count must match tokenIds length");
        assertTrue(result.getGenerationTimeMs() > 0, "Generation time must be positive");
        assertTrue(result.getTokensPerSecond() > 0, "Tokens/sec must be positive");
        assertNotNull(result.getFinishReason(), "Finish reason must not be null");

        log.info("=== Text Generation Result ===");
        log.info("Prompt: {}", prompt);
        log.info("Output: {}", result.getText());
        log.info("Tokens: {}, Time: {}ms, Speed: {:.1f} tok/s",
                result.getGeneratedTokenCount(), result.getGenerationTimeMs(), result.getTokensPerSecond());
        log.info("First token latency: {}ms", result.getFirstTokenLatencyMs());
        log.info("Finish reason: {}", result.getFinishReason());
        log.info("Pipeline: download={}ms import={}ms tokenizer={}ms generate={}ms",
                downloadMs, importMs, tokMs, genMs);

        // Cleanup
        tokenizer.close();
        model.close();
    }

    @Test
    @DisplayName("Text GenerationPipeline: streaming callback")
    public void testTextGenerationStreaming() throws Exception {
        String sizeLabel = System.getProperty("text.model.size", "0.8B");
        String quantStr = System.getProperty("text.quant", "Q4_K_M");
        int maxTokens = Integer.parseInt(System.getProperty("text.max.tokens", "10"));

        LLMModel llmModel = LLMModel.fromSizeLabel(sizeLabel);
        QuantType quantType = QuantType.valueOf(quantStr);

        // Download + import
        DownloadResult downloadResult = LLMModelDownloader.download(llmModel, quantType);
        SameDiff model = GGMLModelImport.importModel(downloadResult.getModelFile().getAbsolutePath());

        String tokenizerUrl = "https://huggingface.co/Qwen/Qwen3.5-" + sizeLabel + "/resolve/main/tokenizer.json";
        File tokenizerFile = LLMModelDownloader.downloadCustom(tokenizerUrl, "qwen35-" + sizeLabel + "-tokenizer.json");
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile.getAbsolutePath());

        model.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        model.setDspAutoCompileEnabled(true);
        model.setDspNativeAutoCompileEnabled(true);

        TextGenerator generator = TextGenerator.builder()
                .model(model)
                .tokenizer(tokenizer)
                .config(SamplingConfig.greedy())
                .build();

        // Collect streamed tokens
        List<String> streamedTokens = new ArrayList<>();
        generator.generateStreaming("Hello, world!", maxTokens, streamedTokens::add);

        log.info("Streamed {} tokens: {}", streamedTokens.size(), String.join("", streamedTokens));
        assertTrue(streamedTokens.size() > 0, "Should stream at least 1 token");

        // Cleanup
        tokenizer.close();
        model.close();
    }

    @Test
    @DisplayName("Text GenerationPipeline: batch generation")
    public void testTextGenerationBatch() throws Exception {
        String sizeLabel = System.getProperty("text.model.size", "0.8B");
        String quantStr = System.getProperty("text.quant", "Q4_K_M");
        int maxTokens = Integer.parseInt(System.getProperty("text.max.tokens", "10"));

        LLMModel llmModel = LLMModel.fromSizeLabel(sizeLabel);
        QuantType quantType = QuantType.valueOf(quantStr);

        // Download + import
        DownloadResult downloadResult = LLMModelDownloader.download(llmModel, quantType);
        SameDiff model = GGMLModelImport.importModel(downloadResult.getModelFile().getAbsolutePath());

        String tokenizerUrl = "https://huggingface.co/Qwen/Qwen3.5-" + sizeLabel + "/resolve/main/tokenizer.json";
        File tokenizerFile = LLMModelDownloader.downloadCustom(tokenizerUrl, "qwen35-" + sizeLabel + "-tokenizer.json");
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile.getAbsolutePath());

        model.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        model.setDspAutoCompileEnabled(true);
        model.setDspNativeAutoCompileEnabled(true);

        TextGenerator generator = TextGenerator.builder()
                .model(model)
                .tokenizer(tokenizer)
                .config(SamplingConfig.greedy())
                .build();

        String[] prompts = {
                "What is 2 + 2?",
                "Name a planet in our solar system."
        };

        GenerationResult[] results = generator.generateBatch(prompts, maxTokens);

        assertNotNull(results, "Batch results must not be null");
        assertEquals(prompts.length, results.length, "Should have one result per prompt");

        for (int i = 0; i < results.length; i++) {
            assertNotNull(results[i], "Result " + i + " must not be null");
            assertNotNull(results[i].getText(), "Result " + i + " text must not be null");
            assertTrue(results[i].getGeneratedTokenCount() > 0,
                    "Result " + i + " should generate at least 1 token");
            log.info("Batch[{}]: {} tokens, {:.1f} tok/s: {}",
                    i, results[i].getGeneratedTokenCount(), results[i].getTokensPerSecond(),
                    results[i].getText());
        }

        // Cleanup
        tokenizer.close();
        model.close();
    }
}
