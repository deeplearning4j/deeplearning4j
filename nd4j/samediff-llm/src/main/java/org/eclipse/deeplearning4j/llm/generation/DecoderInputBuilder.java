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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
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
 * @see GenerationPipeline
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
                // Merged-decoder prefill (use_cache_branch=false): ONNX If semantics
                // never evaluate the with-past branch, so past inputs are DEAD. Feeding
                // empty arrays sends len-0 values through the inactive frame's ops
                // (shape math reads them as real -> OOB/garbage); OMITTING the
                // placeholder yields null, which the interpreted engine's
                // null-propagation skips cleanly until the Merge picks the live branch.
                boolean mergedGraphPrefill = !usingStaticKv
                        && ioConfig.getUseCacheBranchName() != null
                        && decoder != null && decoder.hasVariable(ioConfig.getUseCacheBranchName());
                if (!mergedGraphPrefill) {
                    buildKvCacheInput(decoderInputMap, inputName, decoder, usingStaticKv, usePadded,
                            staticKvBuffers, cachePos, hiddenSize);
                }
            } else if (ioConfig.isUseCacheBranch(inputName)) {
                // Merged-decoder branch selector: prefill (no past yet) runs the
                // no-past branch (false); KV-carrying decode steps run the
                // with-past branch (true).
                decoderInputMap.put(inputName, Nd4j.scalar(usingStaticKv));
            } else if (ioConfig.isEncoderHiddenStates(inputName)) {
                if (encoderOutputs != null) {
                    decoderInputMap.put(inputName, encoderOutputs);
                }
            } else if (ioConfig.isEncoderAttentionMask(inputName)) {
                if (encoderAttentionMask != null) {
                    decoderInputMap.put(inputName, encoderAttentionMask);
                } else if (encoderOutputs != null) {
                    long encoderSeqLen = encoderOutputs.size(1);
                    // Reuse the encoder attention mask — it's always ones(1, encoderSeqLen) with
                    // the same shape every decode step. Fresh allocation each step changes the
                    // specialBuffer() pointer, preventing CUDA graph fast replay (argTableStable).
                    if (canReuse && reusableInputs.containsKey(inputName)) {
                        decoderInputMap.put(inputName, reusableInputs.get(inputName));
                    } else {
                        INDArray mask = Nd4j.ones(DataType.LONG, 1, encoderSeqLen);
                        if (canReuse) reusableInputs.put(inputName, mask);
                        decoderInputMap.put(inputName, mask);
                    }
                }
            } else if (inputName.equals(ioConfig.getAttnMaskReformatOutput())) {
                // Only override for multi-token decode (speculative, seqLen>1).
                // For single-token decode (seqLen==1), the model's internal subgraph
                // computes the causal bias correctly. Injecting an external override
                // corrupts attention weights and produces wrong tokens (loc_0 vs loc_1).
                if (currentSeqLen > 1) {
                    buildAttnMaskReformatOverride(ioConfig, decoderInputMap, inputName, canReuse, usePadded,
                            reusableInputs, maxKvLen, currentSeqLen, cachePos);
                }
            }
        }

        String embedsName = ioConfig.getInputEmbeddingsName();
        if (embedsName != null && !decoderInputMap.containsKey(embedsName)
                && (decoder == null || decoder.hasVariable(embedsName))) {
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
                // Only associate with VARIABLE or CONSTANT type SDVariables.
                // ARRAY-type variables are computed intermediates — calling
                // associateArrayWithVariable on them throws UnsupportedOperationException.
                // PLACEHOLDERS must be excluded too (this method's contract is
                // NON-placeholder inputs): associating one writes placeholdersPerThread,
                // which SHADOWS per-call values on every later run — a warmup's
                // use_cache_branch=true silently overrode prefill's false, activating
                // the with-past If branch during prefill. The externalInputs() check
                // alone misses placeholders under control-flow dependency tracking.
                SDVariable sdVar = decoder.getVariable(inputName);
                if (sdVar != null && sdVar.getVariableType() != VariableType.ARRAY
                        && sdVar.getVariableType() != VariableType.PLACEHOLDER) {
                    decoder.associateArrayWithVariable(arr, inputName);
                }
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
                // Delta update: previous step already set [0..cachePos-2]=1 and
                // [totalSeqLen-1]=1.  Just mark the newly cached position.
                if (cachePos > 0) {
                    mask.putScalar(new long[]{0, cachePos - 1}, 1);
                }
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
            boolean reused = false;
            if (canReuse && reusableInputs.containsKey(inputName)) {
                causalMask = reusableInputs.get(inputName);
                reused = true;
            } else {
                causalMask = Nd4j.zeros(DataType.FLOAT, 1, 1, currentSeqLen, totalSeqLen);
                if (canReuse) reusableInputs.put(inputName, causalMask);
            }
            if (currentSeqLen == 1) {
                if (reused) {
                    // Delta update: the buffer already has the correct state from the
                    // previous step — positions [0..cachePos-1] = 0.0f and
                    // [cachePos..totalSeqLen-1] = MASK_FILL.  Just unmask position cachePos.
                    causalMask.putScalar(new long[]{0, 0, 0, cachePos}, 0.0f);
                } else {
                    // First call: full fill then unmask prefix.
                    causalMask.assign(ModelIOConfig.MASK_FILL);
                    if (cachePos + 1 > 0) {
                        causalMask.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(0, cachePos + 1)).assign(0.0f);
                    }
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
        addConfiguredInputIfInternal(materializedInputNames, decoder, ioConfig.getAttnMaskReformatOutput());

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

    // ========== Scoring / Teacher-Forcing Input Map ==========

    /**
     * Build a complete decoder input map for teacher-forcing, scoring, or perplexity evaluation
     * over any imported decoder, including hybrid architectures with recurrent states (GDN/SSM/conv).
     *
     * <p>Extends {@link #buildDecoderInputMap} with three additional input categories that the
     * standard per-step builder skips (not needed in the decode pipeline but required for
     * single-forward teacher-forcing passes):
     * <ol>
     *   <li><b>Recurrent state inputs</b> (e.g. {@code past_gdn_state.N}, {@code past_conv_state.N}
     *       on hybrid architectures like Qwen3.5 or LFM-2) — fed as zero tensors with shapes
     *       derived from the ops that consume them via
     *       {@link GenerationPipeline#deriveRecurrentStateShape}.</li>
     *   <li><b>position_offset</b> — scalar INT64 = 0 (GGUF in-graph KV models only).</li>
     *   <li><b>cache_position</b> — scalar INT64 = 0 (GGUF in-graph KV models only).</li>
     * </ol>
     *
     * <p>This is the canonical entry point for all scoring uses: perplexity evaluation,
     * distillation target extraction, and any other teacher-forcing forward over an imported
     * decoder. Works for pure attention models, GGUF in-graph KV cache models, and hybrid
     * attention+SSM architectures.
     *
     * <p>Resolve the logits output name separately via
     * {@link ModelIOConfig#findLogitsOutputName(SameDiff)}.
     *
     * @param decoder  the imported SameDiff decoder model
     * @param inputIds input token IDs, shape [1, seqLen]
     * @return complete input map ready for {@code decoder.output(inputs, logitsName)}
     */
    public static Map<String, INDArray> buildScoringInputMap(SameDiff decoder, INDArray inputIds) {
        return buildScoringInputMap(decoder, inputIds, 0L);
    }

    /**
     * Build a complete decoder input map for scoring, with explicit hidden size for KV cache inference.
     *
     * <p>{@code hiddenSize} is used only as a fallback when the KV cache input variable lacks
     * shape information. For most imported GGUF models the shape is embedded in the graph, so
     * passing {@code 0} is safe for input-ids-driven scoring.</p>
     *
     * @param decoder    the imported SameDiff decoder model
     * @param inputIds   input token IDs, shape [1, seqLen]
     * @param hiddenSize model hidden size; pass {@code 0} to infer from the graph
     * @return complete input map ready for {@code decoder.output(inputs, logitsName)}
     */
    public static Map<String, INDArray> buildScoringInputMap(
            SameDiff decoder, INDArray inputIds, long hiddenSize) {

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        long seqLen = inputIds.size(inputIds.rank() - 1);

        // Standard inputs: input_ids, attention_mask, causal_mask, position_ids, empty KV caches.
        // usingStaticKv=false, pastSeqLen=0 → prefill semantics (full causal mask, all-ones attention mask).
        Map<String, INDArray> inputs = buildDecoderInputMap(
                ioConfig, decoder.inputs(), decoder,
                /*embeddings=*/null, inputIds,
                /*pastSeqLen=*/0L, /*currentSeqLen=*/seqLen,
                /*staticKvBuffers=*/null, /*maxKvLen=*/seqLen,
                /*cachePos=*/0L, /*usingStaticKv=*/false, hiddenSize,
                /*reusableInputs=*/null, /*dspActive=*/false,
                /*encoderOutputs=*/null, /*encoderAttentionMask=*/null);

        // Feed position_offset and cache_position scalars (GGUF in-graph KV models).
        // These are not iterated in the main builder loop, so they must be added here.
        String posOffset = ioConfig.getPositionOffsetName();
        if (posOffset != null && decoder.hasVariable(posOffset) && !inputs.containsKey(posOffset)) {
            inputs.put(posOffset, Nd4j.scalar(DataType.INT64, 0));
        }
        String cachePosName = ioConfig.getCachePositionName();
        if (cachePosName != null && decoder.hasVariable(cachePosName) && !inputs.containsKey(cachePosName)) {
            inputs.put(cachePosName, Nd4j.scalar(DataType.INT64, 0));
        }

        // Feed zero-filled recurrent state inputs (hybrid architectures: GDN/SSM/causal-conv).
        // Discovered structurally from the graph op topology (no hardcoded prefix matching).
        List<ModelIOConfig.RecurrentStatePair> recurrentPairs =
                ModelIOConfig.findRecurrentStatePairs(decoder, ioConfig);
        for (ModelIOConfig.RecurrentStatePair pair : recurrentPairs) {
            if (decoder.hasVariable(pair.inputName) && !inputs.containsKey(pair.inputName)) {
                DataType dt = decoder.getVariable(pair.inputName).dataType();
                long[] stateShape = GenerationPipeline.deriveRecurrentStateShape(decoder, pair.inputName);
                if (stateShape != null) {
                    inputs.put(pair.inputName, Nd4j.zeros(dt, stateShape));
                } else {
                    log.warn("buildScoringInputMap: cannot derive shape for recurrent state '{}'; "
                            + "this input will be missing — forward pass may fail", pair.inputName);
                }
            }
        }

        return inputs;
    }

    // ========== In-Graph KV Cache Masks (GGUF models) ==========

    /**
     * Build the attention bias mask for prefill with in-graph KV cache.
     *
     * <p>Shape: [1, 1, prefillLen, maxKvLen]. Lower-triangular causal structure
     * for positions 0..prefillLen-1, with a large negative value for padding beyond prefillLen
     * (dtype-safe: -65504 for FP16, -1e9 for FP32).</p>
     *
     * <pre>
     * Example (prefillLen=3, maxKvLen=6):
     *   [[ 0.0,  -1e9, -1e9, -1e9, -1e9, -1e9],   // position 0 sees only position 0
     *    [ 0.0,  0.0,  -1e9, -1e9, -1e9, -1e9],   // position 1 sees 0..1
     *    [ 0.0,  0.0,  0.0,  -1e9, -1e9, -1e9]]   // position 2 sees 0..2
     * </pre>
     *
     * @param prefillLen number of tokens in the prefill prompt
     * @param maxKvLen total KV buffer size (prefillLen + maxNewTokens)
     * @return attention bias [1, 1, prefillLen, maxKvLen]
     */
    public static INDArray buildInGraphCausalMask(long prefillLen, long maxKvLen, DataType dtype) {
        int Q = (int) prefillLen;
        int K = (int) maxKvLen;
        // Use dtype-safe mask value: -1e9 overflows to -inf in FP16, which causes
        // NaN via (-inf) - (-inf) in softmax max-subtraction. Use -65504 for HALF.
        float maskVal = (dtype == DataType.HALF || dtype == DataType.FLOAT16) ? -65504.0f : -1e9f;
        float[] data = new float[Q * K];

        for (int q = 0; q < Q; q++) {
            int rowOffset = q * K;
            // Mask positions after the current query position AND all padding
            for (int k = q + 1; k < K; k++) {
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
     * Build the attention bias mask for a single decode step with in-graph KV cache.
     *
     * <p>Shape: [1, 1, 1, maxKvLen]. Positions 0..cachePos are unmasked (0.0),
     * positions cachePos+1..maxKvLen-1 are masked (dtype-safe: -65504 for FP16, -1e9 for FP32).</p>
     *
     * @param cachePos current write position in KV buffer (also the query position)
     * @param maxKvLen total KV buffer size
     * @return attention bias [1, 1, 1, maxKvLen]
     */
    public static INDArray buildInGraphDecodeMask(long cachePos, long maxKvLen, DataType dtype) {
        int K = (int) maxKvLen;
        // Use dtype-safe mask value: -1e9 overflows to -inf in FP16, which causes
        // NaN via (-inf) - (-inf) in softmax max-subtraction. Use -65504 for HALF.
        float maskVal = (dtype == DataType.HALF || dtype == DataType.FLOAT16) ? -65504.0f : -1e9f;
        float[] data = new float[K];

        // Unmask positions 0..cachePos (the current token sees all filled positions)
        // Mask positions cachePos+1..maxKvLen-1 (zero-padded cache slots)
        for (int k = (int) cachePos + 1; k < K; k++) {
            data[k] = maskVal;
        }

        INDArray mask = Nd4j.create(data, new long[]{1, 1, 1, maxKvLen}, 'c');
        if (dtype != DataType.FLOAT) {
            INDArray cast = mask.castTo(dtype);
            mask.close();
            return cast;
        }
        return mask;
    }
}
