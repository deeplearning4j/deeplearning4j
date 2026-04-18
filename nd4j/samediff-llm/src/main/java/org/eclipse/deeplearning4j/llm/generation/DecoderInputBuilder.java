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
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Builds the complete decoder input map for each decode step.
 *
 * <p>Handles both dynamic KV (prefill, step 0) and static KV (decode, steps 1+) modes.
 * Constructs inputs_embeds, attention_mask, _causal_mask, input_ids, position_ids,
 * and past_key_values entries based on the decoder's declared input names and
 * a {@link ModelIOConfig} for variable name resolution.</p>
 *
 * <p>Supports optional input reuse (allocate once, update in-place) and DSP padded mode
 * (fixed shapes for CUDA graph capture).</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see ModelIOConfig
 * @see StaticKvCacheDecodeLoop
 */
@Slf4j
public class DecoderInputBuilder {

    /**
     * Build the complete decoder input map for one decode step.
     */
    public static Map<String, INDArray> buildDecoderInputMap(
            List<String> decoderInputNames, SameDiff decoder,
            INDArray embeddings, INDArray inputIds,
            long pastSeqLen, long currentSeqLen,
            Map<String, INDArray> staticKvBuffers, long maxKvLen, long cachePos,
            boolean usingStaticKv, long hiddenSize) {
        return buildDecoderInputMap(decoderInputNames, decoder, embeddings, inputIds,
                pastSeqLen, currentSeqLen, staticKvBuffers, maxKvLen, cachePos,
                usingStaticKv, hiddenSize, null, false);
    }

    /**
     * Build the complete decoder input map with optional reusable input cache.
     */
    public static Map<String, INDArray> buildDecoderInputMap(
            List<String> decoderInputNames, SameDiff decoder,
            INDArray embeddings, INDArray inputIds,
            long pastSeqLen, long currentSeqLen,
            Map<String, INDArray> staticKvBuffers, long maxKvLen, long cachePos,
            boolean usingStaticKv, long hiddenSize,
            Map<String, INDArray> reusableInputs,
            boolean dspActive) {
        ModelIOConfig config = ModelIOConfig.builder().build();
        return buildDecoderInputMap(config, decoderInputNames, decoder, embeddings, inputIds,
                pastSeqLen, currentSeqLen, staticKvBuffers, maxKvLen, cachePos,
                usingStaticKv, hiddenSize, reusableInputs, dspActive,
                null, null);
    }

    /**
     * Build the complete decoder input map using a {@link ModelIOConfig}.
     */
    public static Map<String, INDArray> buildDecoderInputMap(
            ModelIOConfig ioConfig,
            List<String> decoderInputNames, SameDiff decoder,
            INDArray embeddings, INDArray inputIds,
            long pastSeqLen, long currentSeqLen,
            Map<String, INDArray> staticKvBuffers, long maxKvLen, long cachePos,
            boolean usingStaticKv, long hiddenSize,
            Map<String, INDArray> reusableInputs,
            boolean dspActive) {
        return buildDecoderInputMap(ioConfig, decoderInputNames, decoder, embeddings, inputIds,
                pastSeqLen, currentSeqLen, staticKvBuffers, maxKvLen, cachePos,
                usingStaticKv, hiddenSize, reusableInputs, dspActive,
                null, null);
    }

    /**
     * Build the complete decoder input map with encoder-decoder support.
     *
     * <p>This is the primary implementation. All other overloads delegate here.
     * Uses the config's name fields instead of hardcoded string comparisons.</p>
     *
     * @param ioConfig the model I/O configuration with variable name mappings
     * @param decoderInputNames the decoder model's input names
     * @param decoder the SameDiff decoder model
     * @param embeddings current step's embeddings [batch, seqLen, hidden]
     * @param inputIds current step's input IDs [batch, seqLen]
     * @param pastSeqLen logical past sequence length (for position_ids / RoPE)
     * @param currentSeqLen current step's sequence length
     * @param staticKvBuffers map of past_key_values input names to static buffers
     * @param maxKvLen total static KV length
     * @param cachePos next write position in static buffer
     * @param usingStaticKv whether we are in static KV mode
     * @param hiddenSize model hidden size
     * @param reusableInputs optional cache map; populated on first use, updated in-place thereafter
     * @param dspActive whether DSP is active (padded mode with fixed shapes for CUDA graphs)
     * @param encoderOutputs encoder hidden states, null for decoder-only
     * @param encoderAttentionMask encoder attention mask, null if not needed
     * @return map of input name to INDArray ready to pass to decoder.output()
     */
    public static Map<String, INDArray> buildDecoderInputMap(
            ModelIOConfig ioConfig,
            List<String> decoderInputNames, SameDiff decoder,
            INDArray embeddings, INDArray inputIds,
            long pastSeqLen, long currentSeqLen,
            Map<String, INDArray> staticKvBuffers, long maxKvLen, long cachePos,
            boolean usingStaticKv, long hiddenSize,
            Map<String, INDArray> reusableInputs,
            boolean dspActive,
            INDArray encoderOutputs, INDArray encoderAttentionMask) {

        Map<String, INDArray> decoderInputMap = new HashMap<>();
        boolean canReuse = reusableInputs != null && usingStaticKv && currentSeqLen == 1;
        boolean usePadded = dspActive && usingStaticKv;
        List<String> materializedInputNames = expandConfiguredInputNames(ioConfig, decoderInputNames, decoder);

        for (String inputName : materializedInputNames) {
            if (ioConfig.isInputEmbeddings(inputName)) {
                decoderInputMap.put(inputName, embeddings);
            } else if (ioConfig.isAttentionMask(inputName)) {
                buildAttentionMask(ioConfig, decoderInputMap, inputName, canReuse, usePadded,
                        usingStaticKv, reusableInputs, maxKvLen, currentSeqLen, cachePos, pastSeqLen);
            } else if (ioConfig.isCausalMask(inputName)) {
                buildCausalMask(ioConfig, decoderInputMap, inputName, canReuse, usePadded,
                        usingStaticKv, reusableInputs, maxKvLen, currentSeqLen, cachePos, pastSeqLen);
            } else if (ioConfig.isInputIds(inputName)) {
                decoderInputMap.put(inputName, inputIds);
            } else if (ioConfig.isPositionIds(inputName)) {
                buildPositionIds(decoderInputMap, inputName, canReuse, reusableInputs,
                        pastSeqLen, currentSeqLen);
            } else if (ioConfig.isKvCacheInput(inputName)) {
                buildKvCacheInput(decoderInputMap, inputName, decoder, usingStaticKv, usePadded,
                        staticKvBuffers, cachePos, hiddenSize);
            } else if (ioConfig.isEncoderHiddenStates(inputName)) {
                if (encoderOutputs != null) {
                    decoderInputMap.put(inputName, encoderOutputs);
                }
            } else if (ioConfig.isEncoderAttentionMask(inputName)) {
                if (encoderAttentionMask != null) {
                    decoderInputMap.put(inputName, encoderAttentionMask);
                } else if (encoderOutputs != null) {
                    long encoderSeqLen = encoderOutputs.size(1);
                    decoderInputMap.put(inputName, Nd4j.ones(DataType.LONG, 1, encoderSeqLen));
                }
            } else if (inputName.equals(ioConfig.getAttnMaskReformatOutput())) {
                buildAttnMaskReformatOverride(ioConfig, decoderInputMap, inputName, canReuse, usePadded,
                        reusableInputs, maxKvLen, currentSeqLen, cachePos);
            }
        }

        String embedsName = ioConfig.getInputEmbeddingsName();
        if (embedsName != null && !decoderInputMap.containsKey(embedsName)) {
            decoderInputMap.put(embedsName, embeddings);
        }

        associateInternalModelInputs(ioConfig, materializedInputNames, decoder, decoderInputMap);
        return decoderInputMap;
    }

    /**
     * Keep internal decoder inputs phase-aligned with the per-step arrays built above.
     */
    public static void associateInternalModelInputs(ModelIOConfig ioConfig,
                                                    List<String> materializedInputNames,
                                                    SameDiff decoder,
                                                    Map<String, INDArray> decoderInputMap) {
        if (decoder == null || decoderInputMap == null || decoderInputMap.isEmpty()) {
            return;
        }

        List<String> externalInputNames = decoder.externalInputs();
        for (String inputName : materializedInputNames) {
            if (inputName == null || !decoder.hasVariable(inputName)) {
                continue;
            }
            INDArray arr = decoderInputMap.get(inputName);
            if (arr != null && !externalInputNames.contains(inputName)) {
                decoder.associateArrayWithVariable(arr, inputName);
            }
        }
    }

    // ========== Private Helpers ==========

    private static void buildAttentionMask(ModelIOConfig ioConfig,
                                            Map<String, INDArray> decoderInputMap, String inputName,
                                            boolean canReuse, boolean usePadded, boolean usingStaticKv,
                                            Map<String, INDArray> reusableInputs,
                                            long maxKvLen, long currentSeqLen, long cachePos, long pastSeqLen) {
        if (usePadded) {
            long totalSeqLen = maxKvLen + currentSeqLen;
            if (canReuse && reusableInputs.containsKey(inputName)) {
                INDArray mask = reusableInputs.get(inputName);
                if (cachePos > 0) {
                    mask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, cachePos)).assign(1);
                }
                mask.putScalar(0, totalSeqLen - 1, 1);
                decoderInputMap.put(inputName, mask);
            } else {
                INDArray mask = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);
                if (cachePos > 0) {
                    mask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, cachePos)).assign(1);
                }
                mask.putScalar(0, totalSeqLen - 1, 1);
                decoderInputMap.put(inputName, mask);
                if (canReuse) reusableInputs.put(inputName, mask);
            }
        } else if (usingStaticKv) {
            long totalSeqLen = cachePos + currentSeqLen;
            decoderInputMap.put(inputName, Nd4j.ones(DataType.LONG, 1, totalSeqLen));
        } else {
            long totalSeqLen = pastSeqLen + currentSeqLen;
            decoderInputMap.put(inputName, Nd4j.ones(DataType.LONG, 1, totalSeqLen));
        }
    }

    private static void buildCausalMask(ModelIOConfig ioConfig,
                                         Map<String, INDArray> decoderInputMap, String inputName,
                                         boolean canReuse, boolean usePadded, boolean usingStaticKv,
                                         Map<String, INDArray> reusableInputs,
                                         long maxKvLen, long currentSeqLen, long cachePos, long pastSeqLen) {
        if (usePadded) {
            long totalSeqLen = maxKvLen + currentSeqLen;
            INDArray causalMask;
            if (canReuse && reusableInputs.containsKey(inputName)) {
                causalMask = reusableInputs.get(inputName);
            } else {
                causalMask = Nd4j.zeros(DataType.FLOAT, 1, 1, currentSeqLen, totalSeqLen);
                if (canReuse) reusableInputs.put(inputName, causalMask);
            }
            if (currentSeqLen == 1) {
                // Bulk fill: set all to MASK_FILL, then unmask [0..cachePos] with a single assign.
                // Replaces O(maxKvLen - cachePos) putScalar JNI calls with 2 bulk ops.
                causalMask.assign(ModelIOConfig.MASK_FILL);
                if (cachePos + 1 > 0) {
                    causalMask.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.interval(0, cachePos + 1)).assign(0.0f);
                }
            } else {
                causalMask.assign(ModelIOConfig.buildCausalMask(currentSeqLen, totalSeqLen));
            }
            decoderInputMap.put(inputName, causalMask);
        } else if (usingStaticKv) {
            long totalSeqLen = cachePos + currentSeqLen;
            decoderInputMap.put(inputName, ModelIOConfig.buildCausalMask(currentSeqLen, totalSeqLen));
        } else {
            long totalSeqLen = pastSeqLen + currentSeqLen;
            decoderInputMap.put(inputName, ModelIOConfig.buildCausalMask(currentSeqLen, totalSeqLen));
        }
    }

    private static void buildPositionIds(Map<String, INDArray> decoderInputMap, String inputName,
                                          boolean canReuse, Map<String, INDArray> reusableInputs,
                                          long pastSeqLen, long currentSeqLen) {
        if (canReuse && reusableInputs.containsKey(inputName)) {
            INDArray posIds = reusableInputs.get(inputName);
            writePositionIds(posIds, pastSeqLen, currentSeqLen);
            decoderInputMap.put(inputName, posIds);
        } else {
            INDArray posIds = Nd4j.create(DataType.LONG, 1, currentSeqLen);
            writePositionIds(posIds, pastSeqLen, currentSeqLen);
            decoderInputMap.put(inputName, posIds);
            if (canReuse) reusableInputs.put(inputName, posIds);
        }
    }

    private static void buildKvCacheInput(Map<String, INDArray> decoderInputMap, String inputName,
                                           SameDiff decoder, boolean usingStaticKv, boolean usePadded,
                                           Map<String, INDArray> staticKvBuffers, long cachePos,
                                           long hiddenSize) {
        if (usingStaticKv) {
            INDArray staticBuf = staticKvBuffers.get(inputName);
            if (staticBuf != null) {
                if (usePadded) {
                    decoderInputMap.put(inputName, staticBuf);
                } else {
                    if (cachePos > 0 && cachePos < staticBuf.size(2)) {
                        INDArray view = staticBuf.get(
                                NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(0, cachePos), NDArrayIndex.all());
                        decoderInputMap.put(inputName, view);
                    } else if (cachePos >= staticBuf.size(2)) {
                        decoderInputMap.put(inputName, staticBuf);
                    } else {
                        decoderInputMap.put(inputName, ModelIOConfig.createEmptyKvCache(decoder, inputName, 1, hiddenSize));
                    }
                }
            } else {
                decoderInputMap.put(inputName, ModelIOConfig.createEmptyKvCache(decoder, inputName, 1, hiddenSize));
            }
        } else {
            decoderInputMap.put(inputName, ModelIOConfig.createEmptyKvCache(decoder, inputName, 1, hiddenSize));
        }
    }

    private static void buildAttnMaskReformatOverride(ModelIOConfig ioConfig,
                                                       Map<String, INDArray> decoderInputMap, String inputName,
                                                       boolean canReuse, boolean usePadded,
                                                       Map<String, INDArray> reusableInputs,
                                                       long maxKvLen, long currentSeqLen, long cachePos) {
        if (!usePadded) return;

        long totalSeqLen = maxKvLen + currentSeqLen;
        if (canReuse && reusableInputs.containsKey(inputName)) {
            INDArray bias = reusableInputs.get(inputName);
            // Bulk unmask [0..cachePos) — replaces O(cachePos) putScalar JNI calls
            if (cachePos > 0) {
                bias.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, cachePos)).assign(0.0f);
            }
            decoderInputMap.put(inputName, bias);
        } else {
            int totalLen = (int) totalSeqLen;
            float[] biasData = new float[(int) currentSeqLen * totalLen];
            for (int q = 0; q < (int) currentSeqLen; q++) {
                int rowOffset = q * totalLen;
                for (int k = (int) cachePos; k < (int) maxKvLen; k++) {
                    biasData[rowOffset + k] = ModelIOConfig.MASK_FILL;
                }
                for (int k = q + 1; k < (int) currentSeqLen; k++) {
                    biasData[rowOffset + (int) maxKvLen + k] = ModelIOConfig.MASK_FILL;
                }
            }
            INDArray bias = Nd4j.create(biasData, new long[]{1, 1, currentSeqLen, totalLen}, 'c');
            decoderInputMap.put(inputName, bias);
            if (canReuse) reusableInputs.put(inputName, bias);
        }
    }

    static List<String> expandConfiguredInputNames(ModelIOConfig ioConfig,
                                                           List<String> decoderInputNames,
                                                           SameDiff decoder) {
        List<String> materializedInputNames = new ArrayList<>();
        if (decoderInputNames != null) {
            materializedInputNames.addAll(decoderInputNames);
        }
        if (decoder == null) {
            return materializedInputNames;
        }

        addConfiguredInputIfInternal(materializedInputNames, decoder, ioConfig.getInputEmbeddingsName());
        addConfiguredInputIfInternal(materializedInputNames, decoder, ioConfig.getInputIdsName());
        addConfiguredInputIfInternal(materializedInputNames, decoder, ioConfig.getAttentionMaskName());
        addConfiguredInputIfInternal(materializedInputNames, decoder, ioConfig.getCausalMaskName());
        addConfiguredInputIfInternal(materializedInputNames, decoder, ioConfig.getPositionIdsName());
        addConfiguredInputIfInternal(materializedInputNames, decoder, ioConfig.getEncoderHiddenStatesName());
        addConfiguredInputIfInternal(materializedInputNames, decoder, ioConfig.getEncoderAttentionMaskName());

        return materializedInputNames;
    }

    private static void addConfiguredInputIfInternal(List<String> inputNames, SameDiff decoder, String inputName) {
        if (decoder == null || inputName == null || inputName.isEmpty()) {
            return;
        }
        if (!decoder.hasVariable(inputName) || inputNames.contains(inputName)) {
            return;
        }
        inputNames.add(inputName);
    }

    private static void writePositionIds(INDArray posIds, long startPos, long length) {
        if (length == 1) {
            posIds.putScalar(0, 0, startPos);
            return;
        }

        for (long j = 0; j < length; j++) {
            posIds.putScalar(0, j, startPos + j);
        }
    }
}
