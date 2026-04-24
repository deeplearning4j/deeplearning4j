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

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Configuration for decoder model input/output variable names.
 *
 * <p>Eliminates hardcoded variable name assumptions throughout the decode pipeline.
 * All components (GenerationPipeline, DraftModelSpeculator, FrozenDecodeStep)
 * use this config instead of comparing against string literals like
 * "inputs_embeds", "attention_mask", etc.</p>
 *
 * <p>Provides a {@link #discover(SameDiff)} factory that auto-discovers names from
 * the SameDiff graph, and a builder for explicit configuration.</p>
 *
 * @see DraftModelSpeculator
 * @see FrozenDecodeStep
 * @see GenerationPipeline
 */
@Slf4j
@Getter
@Builder
public class ModelIOConfig {

    /** Mask fill value matching torch.finfo(torch.float32).min */
    public static final float MASK_FILL = -3.4028235e+38f;

    /**
     * Holds lists of present key and value output names from a decoder model.
     */
    public static class KVCacheNames {
        public final List<String> keyNames;
        public final List<String> valueNames;

        public KVCacheNames(List<String> keyNames, List<String> valueNames) {
            this.keyNames = keyNames;
            this.valueNames = valueNames;
        }
    }

    /**
     * Find the logits output variable name from a decoder model.
     */
    public static String findLogitsOutputName(SameDiff decoder) {
        for (String outputName : decoder.outputs()) {
            if (outputName.contains("logit") || outputName.equals("logits")) {
                return outputName;
            }
        }
        if (!decoder.outputs().isEmpty()) {
            return decoder.outputs().get(0);
        }
        return null;
    }

    /**
     * Find the KV cache output names (present key/value) from a decoder model.
     */
    public static KVCacheNames findKVCacheOutputNames(SameDiff decoder) {
        List<String> presentKeyNames = new ArrayList<>();
        List<String> presentValueNames = new ArrayList<>();

        for (String outputName : decoder.outputs()) {
            if (outputName.contains("present") && outputName.contains("key")) {
                presentKeyNames.add(outputName);
            } else if (outputName.contains("present") && outputName.contains("value")) {
                presentValueNames.add(outputName);
            }
        }

        Collections.sort(presentKeyNames);
        Collections.sort(presentValueNames);

        return new KVCacheNames(presentKeyNames, presentValueNames);
    }

    /**
     * Build a causal attention mask for the decoder (FLOAT dtype).
     *
     * For prefill (currentSeqLen &gt; 1): upper-triangular mask filled with MASK_FILL
     *   mask[q][k] = 0 if k &lt;= (pastSeqLen + q), else MASK_FILL
     *
     * For decode (currentSeqLen == 1): all zeros (single token attends to all past)
     *
     * @param currentSeqLen number of query tokens in this step
     * @param totalSeqLen total KV length (past + current)
     * @return INDArray of shape [1, 1, currentSeqLen, totalSeqLen] with FLOAT dtype
     */
    public static INDArray buildCausalMask(long currentSeqLen, long totalSeqLen) {
        return buildCausalMask(1, currentSeqLen, totalSeqLen);
    }

    /**
     * Build a batched causal attention mask for the decoder (FLOAT dtype).
     *
     * @param batchSize number of sequences in the batch
     * @param currentSeqLen number of query tokens in this step
     * @param totalSeqLen total KV length (past + current)
     * @return INDArray of shape [batchSize, 1, currentSeqLen, totalSeqLen] with FLOAT dtype
     */
    public static INDArray buildCausalMask(long batchSize, long currentSeqLen, long totalSeqLen) {
        if (currentSeqLen == 1) {
            return Nd4j.zeros(DataType.FLOAT, batchSize, 1, 1, totalSeqLen);
        }

        long pastSeqLen = totalSeqLen - currentSeqLen;
        int Q = (int) currentSeqLen;
        int K = (int) totalSeqLen;
        float[] data = new float[Q * K];

        for (int q = 0; q < Q; q++) {
            int lastVisibleK = (int) pastSeqLen + q;
            int rowOffset = q * K;
            for (int k = lastVisibleK + 1; k < K; k++) {
                data[rowOffset + k] = MASK_FILL;
            }
        }

        INDArray singleMask = Nd4j.create(data, new long[]{1, 1, currentSeqLen, totalSeqLen}, 'c');

        if (batchSize == 1) {
            return singleMask;
        }

        INDArray mask = Nd4j.tile(singleMask, (int) batchSize, 1, 1, 1);
        singleMask.close();
        return mask;
    }

    /**
     * Create an empty KV cache tensor for the first decoder step.
     * Shape is [batchSize, numHeads, 0, headDim].
     *
     * Infers numHeads and headDim from the decoder graph's variable shapes.
     *
     * @param decoder the SameDiff decoder model
     * @param inputName the past_key_values input name
     * @param batchSize batch size
     * @param hiddenSize model hidden size (used for fallback inference)
     * @return empty KV cache INDArray
     */
    public static INDArray createEmptyKvCache(SameDiff decoder, String inputName, long batchSize, long hiddenSize) {
        long numHeads = -1;
        long headDim = -1;
        DataType kvType = DataType.FLOAT;

        SDVariable inputVar = decoder.getVariable(inputName);
        if (inputVar != null && inputVar.getShape() != null && inputVar.getShape().length >= 4) {
            long[] shape = inputVar.getShape();
            if (inputVar.dataType() != null) {
                kvType = inputVar.dataType();
            }
            if (shape[1] > 0) {
                numHeads = shape[1];
            }
            if (shape[3] > 0) {
                headDim = shape[3];
            }
        }

        if (numHeads <= 0 || headDim <= 0) {
            ModelIOConfig defaultConfig = ModelIOConfig.builder().build();
            String presentName = defaultConfig.inputToPresentName(inputName);
            SDVariable presentVar = decoder.getVariable(presentName);
            if (presentVar != null && presentVar.getShape() != null && presentVar.getShape().length >= 4) {
                long[] shape = presentVar.getShape();
                if (numHeads <= 0 && shape[1] > 0) {
                    numHeads = shape[1];
                }
                if (headDim <= 0 && shape[3] > 0) {
                    headDim = shape[3];
                }
            }
        }

        if (headDim <= 0 && numHeads > 0 && hiddenSize > 0) {
            headDim = Math.max(1, hiddenSize / numHeads);
        }
        if (numHeads <= 0 && headDim > 0 && hiddenSize > 0) {
            numHeads = Math.max(1, hiddenSize / headDim);
        }
        if (headDim <= 0) {
            headDim = 64;
        }
        if (numHeads <= 0) {
            numHeads = Math.max(1, hiddenSize / headDim);
        }

        return Nd4j.zeros(kvType, batchSize, numHeads, 0, headDim);
    }

    // ========== Input Names ==========

    /** Name of the input embeddings placeholder (e.g., "inputs_embeds"). */
    @Builder.Default
    private final String inputEmbeddingsName = "inputs_embeds";

    /** Name of the input IDs placeholder (e.g., "input_ids"). May be null if model uses embeddings only. */
    @Builder.Default
    private final String inputIdsName = "input_ids";

    /** Name of the attention mask placeholder (e.g., "attention_mask"). May be null if not used. */
    @Builder.Default
    private final String attentionMaskName = "attention_mask";

    /** Name of the causal mask placeholder (e.g., "_causal_mask"). May be null if not used. */
    @Builder.Default
    private final String causalMaskName = "_causal_mask";

    /** Name of the position IDs placeholder (e.g., "position_ids"). May be null if not used. */
    @Builder.Default
    private final String positionIdsName = "position_ids";

    /**
     * Prefix for KV cache input variables (e.g., "past_key_values.").
     * Used to identify which inputs are KV cache entries vs. regular inputs.
     */
    @Builder.Default
    private final String kvCachePrefix = "past_key_values.";

    /**
     * String replacement pair for mapping present KV output names to past KV input names.
     * Element 0 is the source substring (in the output name), element 1 is the replacement.
     * Default: {"present", "past_key_values"} — e.g., "present.0.key" maps to "past_key_values.0.key".
     */
    @Builder.Default
    private final String[] kvPresentToInputReplace = new String[]{"present", "past_key_values"};

    // ========== Output Names ==========

    /** Name of the logits output variable (e.g., "logits"). */
    @Builder.Default
    private final String logitsOutputName = "logits";

    /**
     * KV cache output names (present key/value pairs).
     * Discovered via {@link #findKVCacheOutputNames(SameDiff)}.
     */
    private final KVCacheNames kvCacheNames;

    // ========== Encoder-Decoder ==========

    /** Name of the encoder hidden states input (for encoder-decoder models like Whisper). */
    @Builder.Default
    private final String encoderHiddenStatesName = "encoder_hidden_states";

    /** Name of the encoder attention mask input (for encoder-decoder models). May be null. */
    @Builder.Default
    private final String encoderAttentionMaskName = "encoder_attention_mask";

    /** Whether this model is an encoder-decoder architecture (e.g., Whisper, T5). */
    @Builder.Default
    private final boolean encoderDecoder = false;

    // ========== Special Nodes ==========

    /**
     * Name of the attn_mask_reformat output node that can be overridden with a placeholder
     * to bypass the subgraph (e.g., for multi-token decode with FrozenDecodeStep).
     * May be null if the model doesn't have this subgraph.
     */
    @Builder.Default
    private final String attnMaskReformatOutput = "/model/attn_mask_reformat/Tile/output_0";

    // ========== Query Methods ==========

    /**
     * Check whether a given input name is the input embeddings variable.
     */
    public boolean isInputEmbeddings(String name) {
        return inputEmbeddingsName != null && inputEmbeddingsName.equals(name);
    }

    /**
     * Check whether a given input name is the input IDs variable.
     */
    public boolean isInputIds(String name) {
        return inputIdsName != null && inputIdsName.equals(name);
    }

    /**
     * Check whether a given input name is the attention mask variable.
     */
    public boolean isAttentionMask(String name) {
        return attentionMaskName != null && attentionMaskName.equals(name);
    }

    /**
     * Check whether a given input name is the causal mask variable.
     */
    public boolean isCausalMask(String name) {
        return causalMaskName != null && causalMaskName.equals(name);
    }

    /**
     * Check whether a given input name is the position IDs variable.
     */
    public boolean isPositionIds(String name) {
        return positionIdsName != null && positionIdsName.equals(name);
    }

    /**
     * Check whether a given input name is a KV cache input.
     */
    public boolean isKvCacheInput(String name) {
        return kvCachePrefix != null && name.startsWith(kvCachePrefix);
    }

    /**
     * Check whether a given input name is a non-KV-cache input that should be cleaned up per step
     * (i.e., not embeddings, input_ids, or KV cache entries).
     */
    public boolean isPerStepDisposableInput(String name) {
        return !isInputEmbeddings(name) && !isInputIds(name) && !isKvCacheInput(name)
                && !isEncoderHiddenStates(name) && !isEncoderAttentionMask(name);
    }

    /**
     * Check whether a given input name is the encoder hidden states variable.
     */
    public boolean isEncoderHiddenStates(String name) {
        return encoderHiddenStatesName != null && encoderHiddenStatesName.equals(name);
    }

    /**
     * Check whether a given input name is the encoder attention mask variable.
     */
    public boolean isEncoderAttentionMask(String name) {
        return encoderAttentionMaskName != null && encoderAttentionMaskName.equals(name);
    }

    /**
     * Check whether a given input name is an encoder-related input.
     */
    public boolean isEncoderInput(String name) {
        return isEncoderHiddenStates(name) || isEncoderAttentionMask(name);
    }

    /**
     * Map a present KV output name to the corresponding past KV input name.
     * E.g., "present.0.key" to "past_key_values.0.key".
     */
    public String presentToInputName(String presentName) {
        if (kvPresentToInputReplace != null && kvPresentToInputReplace.length == 2) {
            return presentName.replace(kvPresentToInputReplace[0], kvPresentToInputReplace[1]);
        }
        return presentName;
    }

    /**
     * Map a past KV input name to the corresponding present KV output name.
     * E.g., "past_key_values.0.key" to "present.0.key".
     * This is the reverse of {@link #presentToInputName(String)}.
     */
    public String inputToPresentName(String inputName) {
        if (kvPresentToInputReplace != null && kvPresentToInputReplace.length == 2) {
            return inputName.replace(kvPresentToInputReplace[1], kvPresentToInputReplace[0]);
        }
        return inputName;
    }

    /**
     * Check whether the model uses input_ids (vs. only inputs_embeds).
     * Determined by whether input_ids is in the decoder's input list.
     */
    public boolean hasInputIds(List<String> decoderInputNames) {
        return inputIdsName != null && decoderInputNames.contains(inputIdsName);
    }

    // ========== Auto-Discovery ==========

    /**
     * Auto-discover a ModelIOConfig from a SameDiff model graph.
     *
     * <p>Inspects the model's input and output variable names to find:
     * <ul>
     *   <li>Input embeddings: first input containing "embed"</li>
     *   <li>Input IDs: first input containing "input_id"</li>
     *   <li>Attention mask: first input containing "attention_mask"</li>
     *   <li>Causal mask: first input containing "causal_mask"</li>
     *   <li>Position IDs: first input containing "position_id"</li>
     *   <li>KV cache prefix: detected from first input starting with common KV prefixes</li>
     *   <li>Logits output: via {@link #findLogitsOutputName(SameDiff)}</li>
     *   <li>KV cache outputs: via {@link #findKVCacheOutputNames(SameDiff)}</li>
     * </ul></p>
     *
     * @param model the SameDiff model to inspect
     * @return a ModelIOConfig with discovered names (with sensible defaults for any not found)
     */
    public static ModelIOConfig discover(SameDiff model) {
        List<String> inputs = model.inputs();

        String embedName = null;
        String idsName = null;
        String maskName = null;
        String causalName = null;
        String posName = null;
        String kvPrefix = null;
        String attnReformat = null;
        String encoderHiddenName = null;
        String encoderMaskName = null;

        for (String input : inputs) {
            // Encoder inputs must be checked first (they also contain "attention_mask" etc.)
            if (encoderHiddenName == null && (input.contains("encoder_hidden") || input.contains("encoder_output"))) {
                encoderHiddenName = input;
            } else if (encoderMaskName == null && input.contains("encoder") && input.contains("attention_mask")) {
                encoderMaskName = input;
            } else if (embedName == null && input.contains("embed")) {
                embedName = input;
            } else if (idsName == null && input.contains("input_id")) {
                idsName = input;
            } else if (maskName == null && input.contains("attention_mask")) {
                maskName = input;
            } else if (causalName == null && input.contains("causal_mask")) {
                causalName = input;
            } else if (posName == null && input.contains("position_id")) {
                posName = input;
            }

            // Detect KV cache prefix
            if (kvPrefix == null) {
                if (input.startsWith("past_key_values.")) {
                    kvPrefix = "past_key_values.";
                } else if (input.startsWith("past_key_value.")) {
                    kvPrefix = "past_key_value.";
                } else if (input.startsWith("pkv.")) {
                    kvPrefix = "pkv.";
                }
            }
        }

        // If input_ids exists as a variable but wasn't found in inputs(),
        // record it so callers can associate correct values before each execution.
        // No graph topology change (no addPlaceholderOverride) — just track the name.
        if (idsName == null && model.hasVariable("input_ids")) {
            idsName = "input_ids";
            log.info("ModelIOConfig.discover(): found 'input_ids' as internal variable (not placeholder). "
                    + "Will associate correct values via associateArrayWithVariable before each step.");
        }

        // Detect attn_mask_reformat output node
        String defaultAttnReformat = "/model/attn_mask_reformat/Tile/output_0";
        if (model.hasVariable(defaultAttnReformat)) {
            attnReformat = defaultAttnReformat;
        } else {
            // Search for alternative patterns
            for (String varName : model.variableMap().keySet()) {
                if (varName.contains("attn_mask_reformat") && varName.contains("output")) {
                    attnReformat = varName;
                    break;
                }
            }
        }

        // Determine present-to-past replacement
        String[] replacement = new String[]{"present", "past_key_values"};
        if (kvPrefix != null && !kvPrefix.equals("past_key_values.")) {
            // For non-standard KV prefixes, try to infer the replacement
            String kvBase = kvPrefix.substring(0, kvPrefix.length() - 1); // Remove trailing dot
            replacement = new String[]{"present", kvBase};
        }

        // Discover outputs
        String logitsName = findLogitsOutputName(model);
        KVCacheNames kvNames = findKVCacheOutputNames(model);

        boolean isEncoderDecoder = encoderHiddenName != null;

        ModelIOConfig config = ModelIOConfig.builder()
                .inputEmbeddingsName(embedName != null ? embedName : "inputs_embeds")
                .inputIdsName(idsName)
                .attentionMaskName(maskName)
                .causalMaskName(causalName)
                .positionIdsName(posName)
                .kvCachePrefix(kvPrefix != null ? kvPrefix : "past_key_values.")
                .kvPresentToInputReplace(replacement)
                .logitsOutputName(logitsName != null ? logitsName : "logits")
                .kvCacheNames(kvNames)
                .encoderHiddenStatesName(encoderHiddenName != null ? encoderHiddenName : "encoder_hidden_states")
                .encoderAttentionMaskName(encoderMaskName)
                .encoderDecoder(isEncoderDecoder)
                .attnMaskReformatOutput(attnReformat)
                .build();

        log.info("ModelIOConfig.discover(): embeddings={}, inputIds={}, attentionMask={}, causalMask={}, " +
                        "positionIds={}, kvPrefix={}, logits={}, kvLayers={}, attnReformat={}, " +
                        "encoderDecoder={}, encoderHidden={}, encoderMask={}",
                config.inputEmbeddingsName, config.inputIdsName, config.attentionMaskName,
                config.causalMaskName, config.positionIdsName, config.kvCachePrefix,
                config.logitsOutputName, kvNames != null ? kvNames.keyNames.size() : 0,
                config.attnMaskReformatOutput,
                config.encoderDecoder, config.encoderHiddenStatesName, config.encoderAttentionMaskName);

        return config;
    }
}
