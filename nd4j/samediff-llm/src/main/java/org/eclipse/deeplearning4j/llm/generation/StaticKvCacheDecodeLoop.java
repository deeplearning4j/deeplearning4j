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
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.TokenSample;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Autoregressive decode loop with static KV cache for CUDA graph capture.
 *
 * <p>Encapsulates the full decode pipeline: prefill with dynamic KV, transition to
 * fixed-shape static KV buffers, frozen shapes for CUDA graph replay, GPU-side
 * token sampling via {@link TokenSample}, and per-step KV scatter.</p>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
 *     .decoder(decoder)
 *     .embedTokens(embedTokens)
 *     .tokenizer(tokenizer)
 *     .maxNewTokens(256)
 *     .hiddenSize(hiddenSize)
 *     .build();
 * GenerationResult result = loop.decode(prefillEmbeddings, promptTokenIds);
 * }</pre>
 */
@Slf4j
@Builder
public class StaticKvCacheDecodeLoop {

    private final SameDiff decoder;
    private final SameDiff embedTokens;
    private final Tokenizer tokenizer;

    @Builder.Default
    private final SamplingConfig samplingConfig = SamplingConfig.greedy();
    @Builder.Default
    private final int maxNewTokens = 256;
    @Builder.Default
    private final long hiddenSize = 0;

    private final String embedInputName;
    private final String[] embedOutputNames;
    private final Set<Integer> additionalStopTokenIds;

    /**
     * Run the autoregressive decode loop.
     *
     * @param prefillEmbeddings merged embeddings for the full prompt [1, seqLen, hidden]
     * @param promptTokenIds the prompt token IDs (used for input_ids at step 0)
     * @return generation result with text, token IDs, and timing
     */
    public GenerationResult decode(INDArray prefillEmbeddings, int[] promptTokenIds) {
        long decodeStart = System.currentTimeMillis();

        // Resolve I/O names
        String logitsOutputName = DecoderUtils.findLogitsOutputName(decoder);
        DecoderUtils.KVCacheNames kvNames = DecoderUtils.findKVCacheOutputNames(decoder);
        List<String> decoderInputNames = decoder.inputs();
        log.info("  Decoder input names: {}", decoderInputNames);

        String resolvedEmbedInputName = embedInputName;
        if (resolvedEmbedInputName == null) {
            resolvedEmbedInputName = embedTokens.inputs().isEmpty() ? "input_ids" : embedTokens.inputs().get(0);
        }
        String[] resolvedEmbedOutputNames = embedOutputNames;
        if (resolvedEmbedOutputNames == null) {
            resolvedEmbedOutputNames = embedTokens.outputs().toArray(new String[0]);
        }

        // Resolve stop tokens
        int eosTokenId = tokenizer.getEosTokenId();
        Set<Integer> stopTokenIds = new HashSet<>();
        stopTokenIds.add(eosTokenId);
        Integer endOfUtteranceTokenId = tokenizer.getTokenId("<end_of_utterance>");
        if (endOfUtteranceTokenId != null) {
            stopTokenIds.add(endOfUtteranceTokenId);
        }
        if (additionalStopTokenIds != null) {
            stopTokenIds.addAll(additionalStopTokenIds);
        }

        // Resolve hidden size
        long resolvedHiddenSize = hiddenSize;
        if (resolvedHiddenSize <= 0) {
            resolvedHiddenSize = prefillEmbeddings.shape()[2];
        }

        // Extract embedding weight table for direct lookup (avoids full SameDiff.output() per token).
        // The weight may be stored as CONSTANT or VARIABLE depending on the ONNX import path.
        INDArray embeddingTable = null;
        for (SDVariable var : embedTokens.variables()) {
            if (var.getVariableType() == VariableType.CONSTANT || var.getVariableType() == VariableType.VARIABLE) {
                INDArray arr = var.getArr();
                if (arr != null && arr.rank() == 2) {
                    if (embeddingTable == null || arr.length() > embeddingTable.length()) {
                        embeddingTable = arr;
                    }
                }
            }
        }
        if (embeddingTable != null) {
            log.info("  Using direct embedding lookup: shape={} (bypasses SameDiff.output() per token)",
                    Arrays.toString(embeddingTable.shape()));
        } else {
            log.warn("  Could not extract embedding table, falling back to SameDiff.output() per token");
        }

        // Build output name list
        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(logitsOutputName);
        allOutputNames.addAll(kvNames.keyNames);
        allOutputNames.addAll(kvNames.valueNames);

        // State
        List<Integer> generatedTokens = new ArrayList<>();
        INDArray currentEmbeddings = prefillEmbeddings;
        INDArray currentInputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length).castTo(DataType.LONG);
        long pastSeqLen = 0;

        // Per-step timing
        List<Long> stepTimesMs = new ArrayList<>();
        long prefillTimeMs = 0;

        // Phase timing accumulators (steps 3+)
        long totalInputBuildNs = 0, totalDecoderNs = 0, totalLogitsDupNs = 0;
        long totalKvUpdateNs = 0, totalSamplingNs = 0;
        int detailSteps = 0;

        // Static KV cache state
        Map<String, INDArray> staticKvBuffers = null;
        long maxKvLen = -1;
        long cachePos = 0;
        boolean usingStaticKv = false;
        boolean kvScatterInCpp = false;  // When true, C++ handles KV scatter — skip Java side

        // Reusable input arrays — avoids per-step allocation of masks/position_ids
        // Also used for CUDA graph replay: fixed-address buffers for inputs_embeds/input_ids
        // prevent address key mismatch that would invalidate captured graphs.
        Map<String, INDArray> reusableInputs = new HashMap<>();
        // Fixed-address decode buffers (allocated once, data copied each step)
        INDArray reusableEmbeddings = null;  // [1, 1, hiddenSize]
        INDArray reusableInputIds = null;    // [1, 1]

        // Custom attention bias for bypassing attn_mask_reformat (enables fixed-shape CUDA graphs)
        Map<String, INDArray> customAttnBias = null;  // null = use view-based approach
        INDArray attnBiasTemplate = null;  // pre-allocated [1,1,1,maxKvLen+1], updated in-place

        GenerationResult.FinishReason finishReason = GenerationResult.FinishReason.MAX_TOKENS;

        // Clear any pre-compiled DSP plan before the prefill step.
        // BenchmarkConfigApplier may have compiled a plan using empty placeholder shapes
        // (past_key_values = [1,3,0,64]), which caches empty present KV shapes. The prefill
        // step needs fresh shape computation based on the actual input shapes.
        {
            InferenceSession prefillCheckSession = decoder.getOrCreateSession();
            DynamicShapePlanExecutor prefillDspCheck = prefillCheckSession.getDynamicShapePlanExecutor();
            if (prefillDspCheck != null && prefillDspCheck.getCurrentPlan() != null) {
                log.info("  Clearing pre-compiled DSP plan for prefill (stale shapes from empty placeholders)");
                decoder.clearDynamicShapePlanCache();
                prefillCheckSession.clearAllCaches();
            }
        }

        for (int step = 0; step < maxNewTokens; step++) {
            long stepStart = System.nanoTime();
            long currentSeqLen = currentEmbeddings.shape()[1];

            // Build input map (with reusable input cache for decode steps)
            // Rebuild custom attn bias each step — the executor may close the previous one.
            // The bias shape [1,1,1,maxKvLen+1] is constant, only values change (gap shrinks).
            if (customAttnBias != null && step >= 1) {
                INDArray freshBias = DecoderUtils.buildStaticKvAttnBias(maxKvLen, cachePos);
                for (String key : customAttnBias.keySet()) {
                    customAttnBias.put(key, freshBias);
                }
            }
            Map<String, INDArray> decoderInputMap = DecoderUtils.buildDecoderInputMap(
                    decoderInputNames, decoder,
                    currentEmbeddings, currentInputIds,
                    pastSeqLen, currentSeqLen,
                    staticKvBuffers, maxKvLen, cachePos,
                    usingStaticKv, resolvedHiddenSize,
                    reusableInputs, customAttnBias);

            long tAfterInputBuild = System.nanoTime();

            // Diagnostic logging for steps 1-3
            if (step >= 1 && step <= 3) {
                logDiagnostics(step, decoderInputMap);
            }

            // Run decoder — use fast path when shapes are frozen (skips setCloseable overhead)
            Map<String, INDArray> decoderOutputs;
            boolean useDirect = usingStaticKv && step >= 2
                    && !"true".equalsIgnoreCase(System.getProperty("nd4j.dsp.noDirect"));
            if (useDirect) {
                decoderOutputs = decoder.outputDirect(
                        decoderInputMap, allOutputNames.toArray(new String[0]));
            } else {
                decoderOutputs = decoder.output(
                        decoderInputMap, allOutputNames.toArray(new String[0]));
            }

            long tAfterDecoder = System.nanoTime();

            // Diagnostic: present KV shapes at steps 0-3 (skip when C++ KV scatter manages outputs)
            if (step <= 3 && !kvScatterInCpp) {
                logPresentKvDiagnostics(step, decoderOutputs, kvNames);
            }

            // Extract logits — keep raw reference, no dup needed
            // TokenSample now accepts rank-3 [batch, seqLen, vocabSize] directly
            INDArray logitsRaw = decoderOutputs.get(logitsOutputName);
            if (logitsRaw == null) {
                log.error("No logits output at step {}", step);
                break;
            }
            // Mixed precision decoder (HALF weights + FLOAT activations) keeps logits in FLOAT.
            // If a future full-HALF path produces HALF logits, cast to FLOAT for sampling precision.
            if (logitsRaw.dataType() == DataType.HALF) {
                INDArray floatLogits = logitsRaw.castTo(DataType.FLOAT);
                if (logitsRaw.closeable()) logitsRaw.close();
                logitsRaw = floatLogits;
            }
            // Note: DSP outputs may have isConstant flags. We clear the stale native
            // error state before token_sample exec instead of calling setCloseable(true),
            // because setCloseable makes the DSP's frozen output closeable, and closing
            // it destroys the DSP's internal output slot cache → fallback to Java path.

            // Diagnostic: top logit values at steps 0-3
            if (step == 0) {
                logLogitsDiagnostics(step, logitsRaw);
            }

            long tAfterLogitsDup = System.nanoTime();

            // KV cache update
            if (usingStaticKv) {
                if (kvScatterInCpp) {
                    // C++ frozen fast path handles scatter + position advance automatically
                    cachePos++;
                } else {
                    // Java fallback: scatter new KV entries via view+assign
                    DecoderUtils.scatterNewKvEntries(staticKvBuffers, decoderOutputs,
                            kvNames.keyNames, kvNames.valueNames, maxKvLen, cachePos);
                    cachePos++;
                }
                // Close present KV outputs — scatter already copied data into static buffers.
                // Without this, 60 tensors × ~4.5MB = ~270MB leaked per step.
                // Skip when C++ KV scatter is active — C++ manages these buffer lifecycles.
                if (!kvScatterInCpp) {
                    int kvClosed = 0;
                    long kvClosedBytes = 0;
                    int kvSkippedClosed = 0, kvSkippedNull = 0;
                    for (String pn : kvNames.keyNames) {
                        INDArray pv = decoderOutputs.get(pn);
                        if (pv == null) { kvSkippedNull++; continue; }
                        if (pv.wasClosed() || pv.data() == null) { kvSkippedClosed++; continue; }
                        long bytes = pv.data().length() * pv.data().getElementSize();
                        pv.setCloseable(true);
                        pv.close();
                        kvClosed++;
                        kvClosedBytes += bytes;
                    }
                    for (String pn : kvNames.valueNames) {
                        INDArray pv = decoderOutputs.get(pn);
                        if (pv == null) { kvSkippedNull++; continue; }
                        if (pv.wasClosed() || pv.data() == null) { kvSkippedClosed++; continue; }
                        long bytes = pv.data().length() * pv.data().getElementSize();
                        pv.setCloseable(true);
                        pv.close();
                        kvClosed++;
                        kvClosedBytes += bytes;
                    }
                    if (step < 5 || step % 10 == 0) {
                        log.info("  [KV-CLOSE] step={} closed={} ({}MB) skippedClosed={} skippedNull={}",
                                step, kvClosed, kvClosedBytes / (1024 * 1024), kvSkippedClosed, kvSkippedNull);
                    }
                }
            } else {
                // Step 0 (prefill): transition to static KV
                long prefillSeqLen = currentSeqLen;
                maxKvLen = prefillSeqLen + maxNewTokens;
                log.info("  Setting up static KV: prefillLen={}, maxKvLen={} ({} tensors)",
                        prefillSeqLen, maxKvLen, kvNames.keyNames.size() + kvNames.valueNames.size());
                staticKvBuffers = DecoderUtils.padKvCacheToStaticSize(
                        decoderOutputs, kvNames.keyNames, kvNames.valueNames, maxKvLen);
                Nd4j.getExecutioner().commit();
                cachePos = prefillSeqLen;
                usingStaticKv = true;

                // Close prefill KV outputs — but ONLY when DSP native executor is NOT active.
                // When DSP is active, the C++ slotArrayCache_ still holds raw NDArray* pointers
                // to these outputs. Java close() calls opaqueNDArray.close() which deletes the
                // C++ NDArray, leaving slotArrayCache_ with dangling pointers → use-after-free
                // at the next execution step. The DSP will evict stale arrays naturally when
                // shapes change (prefill [1,h,679,d] → decode [1,h,699,d]).
                InferenceSession prefillSession = decoder.getOrCreateSession();
                boolean dspActive = prefillSession.getDynamicShapePlanExecutor() != null
                        && prefillSession.getDynamicShapePlanExecutor().getCurrentPlan() != null;
                if (!dspActive) {
                    for (String pn : kvNames.keyNames) {
                        INDArray pv = decoderOutputs.get(pn);
                        if (pv != null) { pv.setCloseable(true); pv.close(); }
                    }
                    for (String pn : kvNames.valueNames) {
                        INDArray pv = decoderOutputs.get(pn);
                        if (pv != null) { pv.setCloseable(true); pv.close(); }
                    }
                }

                // Detect attn_mask_reformat subgraph outputs in the decoder graph.
                // If found, we can pre-compute the attention bias in Java and pass the
                // full static KV buffer (fixed shapes → CUDA graph capture). Without it,
                // fall back to view-based approach (variable shapes, no CUDA graphs).
                List<String> attnMaskVars = DecoderUtils.findAttnMaskReformatOutputs(decoder);
                if (!attnMaskVars.isEmpty()) {
                    // Custom bias mode: bypass attn_mask_reformat, use full static buffer
                    attnBiasTemplate = DecoderUtils.buildStaticKvAttnBias(maxKvLen, cachePos);
                    customAttnBias = new HashMap<>();
                    for (String varName : attnMaskVars) {
                        customAttnBias.put(varName, attnBiasTemplate);
                    }
                    log.info("  [BIAS-OVERRIDE] Detected {} attn_mask_reformat output(s): {}",
                            attnMaskVars.size(), attnMaskVars);
                    log.info("  [BIAS-OVERRIDE] Using pre-computed bias [1,1,1,{}] — full static KV, fixed shapes",
                            maxKvLen + 1);

                    // Freeze shapes for CUDA graph capture
                    InferenceSession decoderSession = decoder.getOrCreateSession();
                    DynamicShapePlanExecutor dspExec = decoderSession.getDynamicShapePlanExecutor();
                    boolean skipFreeze = "true".equalsIgnoreCase(System.getProperty("nd4j.dsp.nofreeze"));
                    if (dspExec != null && !skipFreeze) {
                        dspExec.setShapesFrozen(true);
                        dspExec.setTraceEnabled(true);
                        dspExec.setExecutionTimingEnabled(true);

                        // C++ KV scatter is disabled for bias-override mode.
                        // The bias-override approach uses static KV buffers where the model's
                        // internal concat produces present=[B,H,maxKvLen+1,D] outputs. The
                        // Java-side scatter correctly extracts the last position (new token)
                        // and copies it into the static past buffer. The C++ KV scatter was
                        // designed for frozen-KV (no-concat) mode and doesn't handle the
                        // concat-based output correctly → stale KV → degenerate output.
                        // TODO: Fix C++ KV scatter for concat-based present outputs.
                        log.info("  [Perf] Using Java-side KV scatter (bias-override mode)");

                        log.info("  [Perf] Shapes frozen — static KV buffer shape=[1,h,{},d], decode fast path active", maxKvLen);
                    } else {
                        log.warn("  [Perf] No DSP executor found to freeze shapes");
                    }
                } else {
                    // No attn_mask_reformat detected — fall back to view-based approach.
                    // Shapes grow each step, no CUDA graph capture possible.
                    InferenceSession decoderSession = decoder.getOrCreateSession();
                    DynamicShapePlanExecutor dspExec = decoderSession.getDynamicShapePlanExecutor();
                    if (dspExec != null && dspExec.getCurrentPlan() != null) {
                        decoder.clearDynamicShapePlanCache();
                        decoderSession.clearAllCaches();
                        log.info("  [KV-VIEW] Cleared DSP plan — shapes grow each step, cannot freeze");
                    }
                    log.info("  [KV-VIEW] No attn_mask_reformat detected — using view approach: past_kv=[0:cachePos], mask=all-ones");
                }
            }

            long tAfterKvUpdate = System.nanoTime();

            // Sample next token via native GPU op — pass logits directly (rank-3 supported)
            long tSampStart = System.nanoTime();

            // Pre-allocate output array to avoid calculateOutputShape on constant ShapeInfo.
            // DSP outputs may have constant ShapeInfo flags that cause calculateOutputShape
            // to fail when token_sample tries to infer output shapes internally.
            long batchSize = logitsRaw.rank() == 3 ? logitsRaw.size(0) :
                    (logitsRaw.rank() == 2 ? logitsRaw.size(0) : 1);
            INDArray tokenOutput = Nd4j.createUninitialized(DataType.INT64, batchSize);

            TokenSample tokenSampleOp;
            if (samplingConfig.isGreedy()) {
                tokenSampleOp = new TokenSample(logitsRaw);
            } else {
                tokenSampleOp = new TokenSample(logitsRaw,
                        samplingConfig.getTemperature(),
                        samplingConfig.getTopK(),
                        samplingConfig.getTopP(),
                        samplingConfig.getSeed() != null ? samplingConfig.getSeed() : 0L);
            }
            tokenSampleOp.addOutputArgument(tokenOutput);
            // Clear stale native error state from DSP graph capture failures.
            // DSP may leave non-zero errorCode after capture failure even though
            // execution succeeded via slot-by-slot fallback.
            Nd4j.getNativeOps().clearLastError();
            INDArray tokenResult = Nd4j.getExecutioner().exec(tokenSampleOp)[0];
            int nextTokenId = tokenResult.getInt(0);
            long tSampArgmax = System.nanoTime();
            generatedTokens.add(nextTokenId);

            long stepElapsedNs = System.nanoTime() - stepStart;
            long stepElapsedMs = stepElapsedNs / 1_000_000;

            // Log sampling sub-timings
            if (step < 6 || step % 10 == 0) {
                log.info("  [SAMP] step={} total={}ms (native token_sample, no view/dup)",
                        step,
                        (tSampArgmax - tSampStart) / 1_000_000);
            }

            // Accumulate detailed timing for steps 3+
            if (step >= 3) {
                totalInputBuildNs += tAfterInputBuild - stepStart;
                totalDecoderNs    += tAfterDecoder - tAfterInputBuild;
                totalLogitsDupNs  += tAfterLogitsDup - tAfterDecoder;
                totalKvUpdateNs   += tAfterKvUpdate - tAfterLogitsDup;
                totalSamplingNs   += stepElapsedNs - (tAfterKvUpdate - stepStart);
                detailSteps++;
            }

            if (step == 0) {
                prefillTimeMs = stepElapsedMs;
                log.info("  Step 0 (prefill): {}ms (seq_len={})", stepElapsedMs, currentSeqLen);
            } else {
                stepTimesMs.add(stepElapsedMs);
            }

            String tokenText = tokenizer.decode(new int[]{nextTokenId}, false);

            // Log every 10 steps or first 6
            if (step < 6 || step % 10 == 0) {
                double currentTokPerSec = step > 0 && stepElapsedMs > 0 ? 1000.0 / stepElapsedMs : 0;
                if (step >= 2) {
                    long inputMs  = (tAfterInputBuild - stepStart) / 1_000_000;
                    long decMs    = (tAfterDecoder - tAfterInputBuild) / 1_000_000;
                    long dupMs    = (tAfterLogitsDup - tAfterDecoder) / 1_000_000;
                    long kvMs     = (tAfterKvUpdate - tAfterLogitsDup) / 1_000_000;
                    long sampMs   = stepElapsedMs - (tAfterKvUpdate - stepStart) / 1_000_000;
                    log.info("  Step {}: '{}' (id={}) {}ms ({} tok/s) [input={}ms dec={}ms dup={}ms kv={}ms samp={}ms cachePos={}]",
                            step, tokenText, nextTokenId, stepElapsedMs, String.format("%.1f", currentTokPerSec),
                            inputMs, decMs, dupMs, kvMs, sampMs, cachePos - 1);
                } else {
                    log.info("  Step {}: '{}' (id={}) {}ms ({} tok/s)",
                            step, tokenText, nextTokenId, stepElapsedMs, String.format("%.1f", currentTokPerSec));
                }
            }

            // Check stop tokens
            if (stopTokenIds.contains(nextTokenId)) {
                log.info("  Stop token at step {}", step);
                finishReason = GenerationResult.FinishReason.EOS;
                // Only close if closeable (zero-copy outputs are non-closeable, managed by DSP cache)
                if (logitsRaw.closeable()) {
                    logitsRaw.close();
                }
                break;
            }

            // Only close if closeable (zero-copy outputs are non-closeable, managed by DSP cache)
            if (logitsRaw.closeable()) {
                logitsRaw.close();
            }

            // Clean up per-step inputs (masks, position_ids — NOT embeddings, input_ids, static KV, or reusable)
            for (var entry : decoderInputMap.entrySet()) {
                String name = entry.getKey();
                INDArray arr = entry.getValue();
                if (name.equals("inputs_embeds") || name.equals("input_ids")) continue;
                if (name.startsWith("past_key_values.")) continue;
                // Skip arrays managed by the reusable inputs cache
                if (reusableInputs.containsValue(arr)) continue;
                if (arr != null && !arr.wasClosed()) {
                    arr.setCloseable(true);
                    arr.close();
                }
            }
            decoder.clearPlaceholders(false);

            // Get embedding for next token
            INDArray prevEmbeddings = currentEmbeddings;
            if (embeddingTable != null) {
                // Direct lookup: single row from embedding table, reshaped to [1, 1, hiddenSize]
                INDArray rowEmbed = embeddingTable.getRow(nextTokenId).reshape(1, 1, resolvedHiddenSize);
                // Use fixed-address buffer for CUDA graph replay stability
                if (usingStaticKv) {
                    if (reusableEmbeddings == null) {
                        reusableEmbeddings = rowEmbed.dup();
                    } else {
                        reusableEmbeddings.assign(rowEmbed);
                    }
                    currentEmbeddings = reusableEmbeddings;
                } else {
                    currentEmbeddings = rowEmbed;
                }
                if (usingStaticKv) {
                    if (reusableInputIds == null) {
                        reusableInputIds = Nd4j.createFromArray(new int[]{nextTokenId}).reshape(1, 1).castTo(DataType.LONG);
                    } else {
                        reusableInputIds.putScalar(0, 0, nextTokenId);
                    }
                    currentInputIds = reusableInputIds;
                } else {
                    INDArray newTokenTensor = Nd4j.createFromArray(new int[]{nextTokenId}).reshape(1, 1).castTo(DataType.LONG);
                    if (currentInputIds != null && currentInputIds != newTokenTensor && !currentInputIds.wasClosed()) {
                        currentInputIds.setCloseable(true);
                        currentInputIds.close();
                    }
                    currentInputIds = newTokenTensor;
                }
            } else {
                // Fallback: full SameDiff execution
                INDArray newTokenTensor = Nd4j.createFromArray(new int[]{nextTokenId}).reshape(1, 1).castTo(DataType.LONG);
                Map<String, INDArray> newEmbedOutputs = embedTokens.output(
                        Map.of(resolvedEmbedInputName, newTokenTensor), resolvedEmbedOutputNames);
                for (var entry : newEmbedOutputs.entrySet()) {
                    if (usingStaticKv) {
                        if (reusableEmbeddings == null) {
                            reusableEmbeddings = entry.getValue().dup();
                        } else {
                            reusableEmbeddings.assign(entry.getValue());
                        }
                        currentEmbeddings = reusableEmbeddings;
                    } else {
                        currentEmbeddings = entry.getValue();
                    }
                }
                if (currentInputIds != null && currentInputIds != newTokenTensor && !currentInputIds.wasClosed()) {
                    currentInputIds.setCloseable(true);
                    currentInputIds.close();
                }
                currentInputIds = newTokenTensor;
                embedTokens.clearPlaceholders(false);
            }
            // Close previous embeddings — but NOT if it's the same object as
            // currentEmbeddings (reusableEmbeddings is updated in-place via assign(),
            // so prevEmbeddings == currentEmbeddings == reusableEmbeddings).
            // Also skip the original prefillEmbeddings — it's externally owned by the caller.
            if (prevEmbeddings != null && prevEmbeddings != currentEmbeddings
                    && prevEmbeddings != prefillEmbeddings
                    && !prevEmbeddings.wasClosed()) {
                prevEmbeddings.setCloseable(true);
                prevEmbeddings.close();
            }
            pastSeqLen += currentSeqLen;
        }

        // Release reusable input arrays
        for (INDArray arr : reusableInputs.values()) {
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        reusableInputs.clear();

        // Release static KV buffers
        if (staticKvBuffers != null) {
            for (INDArray buf : staticKvBuffers.values()) {
                if (buf != null && !buf.wasClosed()) { buf.setCloseable(true); buf.close(); }
            }
        }

        long totalDecodeMs = System.currentTimeMillis() - decodeStart;

        // Log phase breakdown
        if (detailSteps > 0) {
            log.info("=== PHASE BREAKDOWN (avg over {} decode steps, fast path steps 3+) ===", detailSteps);
            log.info("  Input build:  {}ms", totalInputBuildNs / detailSteps / 1_000_000);
            log.info("  Decoder exec: {}ms", totalDecoderNs    / detailSteps / 1_000_000);
            log.info("  Logits dup:   {}ms", totalLogitsDupNs  / detailSteps / 1_000_000);
            log.info("  KV update:    {}ms", totalKvUpdateNs   / detailSteps / 1_000_000);
            log.info("  Sampling:     {}ms", totalSamplingNs   / detailSteps / 1_000_000);
            log.info("  Sum:          {}ms", (totalInputBuildNs + totalDecoderNs + totalLogitsDupNs + totalKvUpdateNs + totalSamplingNs) / detailSteps / 1_000_000);
        }

        // Build result
        int[] tokenIds = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
        String generatedText = tokenizer.decode(tokenIds, false);

        int decodeTokens = stepTimesMs.size();
        double avgDecodeMs = decodeTokens > 0
                ? stepTimesMs.stream().mapToLong(Long::longValue).average().orElse(0) : 0;
        double decodeTokensPerSec = avgDecodeMs > 0 ? 1000.0 / avgDecodeMs : 0;
        long p50Ms = decodeTokens > 0 ? percentile(stepTimesMs, 50) : 0;
        long p90Ms = decodeTokens > 0 ? percentile(stepTimesMs, 90) : 0;
        long p99Ms = decodeTokens > 0 ? percentile(stepTimesMs, 99) : 0;

        log.info("========================================");
        log.info("GENERATED TEXT ({} tokens):", generatedTokens.size());
        log.info("{}", generatedText);
        log.info("========================================");
        log.info("PERFORMANCE SUMMARY:");
        log.info("  Prefill (step 0):  {}ms", prefillTimeMs);
        log.info("  Decode tokens:     {} (excluding prefill)", decodeTokens);
        log.info("  Avg decode time:   {}ms/token", String.format("%.1f", avgDecodeMs));
        log.info("  Decode throughput: {} tok/s", String.format("%.2f", decodeTokensPerSec));
        log.info("  Latency P50/P90/P99: {}ms / {}ms / {}ms", p50Ms, p90Ms, p99Ms);
        log.info("  Total decode time: {}ms ({} tokens)", totalDecodeMs, generatedTokens.size());
        log.info("========================================");

        if (finishReason == GenerationResult.FinishReason.EOS) {
            return GenerationResult.eos(generatedText, tokenIds, promptTokenIds.length,
                    prefillTimeMs, totalDecodeMs);
        } else {
            return GenerationResult.maxTokens(generatedText, tokenIds, promptTokenIds.length,
                    prefillTimeMs, totalDecodeMs);
        }
    }

    private void logDiagnostics(int step, Map<String, INDArray> decoderInputMap) {
        for (var entry : decoderInputMap.entrySet()) {
            INDArray v = entry.getValue();
            String name = entry.getKey();
            if (name.equals("_causal_mask")) {
                log.info("  [DIAG] step={} {}: shape={} min={} max={} nonzero={}",
                        step, name, Arrays.toString(v.shape()),
                        v.minNumber().floatValue(), v.maxNumber().floatValue(),
                        v.neq(0).sumNumber().longValue());
                long len = v.length();
                INDArray flat = v.reshape(len);
                StringBuilder sb = new StringBuilder("  [DIAG]   values[0..4]=");
                for (int i = 0; i < Math.min(5, len); i++) sb.append(flat.getFloat(i)).append(",");
                sb.append(" ... values[").append(len - 5).append("..").append(len - 1).append("]=");
                for (long i = Math.max(0, len - 5); i < len; i++) sb.append(flat.getFloat(i)).append(",");
                log.info(sb.toString());
            } else if (name.equals("attention_mask")) {
                log.info("  [DIAG] step={} {}: shape={} sum={}",
                        step, name, Arrays.toString(v.shape()), v.sumNumber().longValue());
            } else if (name.startsWith("past_key_values.") && name.contains(".key.0")) {
                log.info("  [DIAG] step={} {}: shape={} absMax={}",
                        step, name, Arrays.toString(v.shape()), v.amaxNumber().floatValue());
            } else if (name.equals("position_ids")) {
                log.info("  [DIAG] step={} {}: shape={} values={}",
                        step, name, Arrays.toString(v.shape()), v);
            }
        }
    }

    private void logPresentKvDiagnostics(int step, Map<String, INDArray> decoderOutputs,
                                          DecoderUtils.KVCacheNames kvNames) {
        for (String pn : kvNames.keyNames) {
            INDArray pv = decoderOutputs.get(pn);
            if (pv != null) {
                if (step == 0 || pn.contains(".0")) {
                    long seqDim = pv.rank() >= 3 ? pv.shape()[2] : -1;
                    if (seqDim > 0) {
                        log.info("  [DIAG] step={} present {}: shape={} isEmpty={} lastPosAbsMax={}",
                                step, pn, Arrays.toString(pv.shape()), pv.isEmpty(),
                                pv.get(NDArrayIndex.point(0), NDArrayIndex.all(),
                                        NDArrayIndex.point(seqDim - 1), NDArrayIndex.all()).amaxNumber().floatValue());
                    } else {
                        log.info("  [DIAG] step={} present {}: shape={} isEmpty={} rank={} seqDim={}",
                                step, pn, Arrays.toString(pv.shape()), pv.isEmpty(), pv.rank(), seqDim);
                    }
                }
            } else if (step == 0) {
                log.info("  [DIAG] step={} present {}: NULL", step, pn);
            }
        }
        if (step == 0) {
            for (String pn : kvNames.valueNames) {
                INDArray pv = decoderOutputs.get(pn);
                if (pv != null && pn.contains(".0")) {
                    long seqDim = pv.rank() >= 3 ? pv.shape()[2] : -1;
                    log.info("  [DIAG] step={} present {}: shape={} isEmpty={} seqDim={}",
                            step, pn, Arrays.toString(pv.shape()), pv.isEmpty(), seqDim);
                }
            }
        }
    }

    private void logLogitsDiagnostics(int step, INDArray logits) {
        try {
            INDArray lastLogitsDiag = logits.rank() == 3
                    ? logits.get(NDArrayIndex.point(0), NDArrayIndex.point(logits.size(1) - 1), NDArrayIndex.all())
                    : logits.getRow(0);
            // Use only getFloat() — CUDA reduction ops (min/max/mean/argmax) fail on
            // constant-flagged DSP output arrays. getFloat() works on any array.
            int len = (int) lastLogitsDiag.length();
            int topId = 0;
            float topVal = lastLogitsDiag.getFloat(0);
            float logitsMin = topVal, logitsMax = topVal;
            double logitsSum = topVal;
            boolean hasNaN = Float.isNaN(topVal);
            for (int i = 1; i < len; i++) {
                float v = lastLogitsDiag.getFloat(i);
                if (Float.isNaN(v)) hasNaN = true;
                if (v > topVal) { topVal = v; topId = i; }
                if (v < logitsMin) logitsMin = v;
                if (v > logitsMax) logitsMax = v;
                logitsSum += v;
            }
            float logitsMean = (float) (logitsSum / len);
            boolean allZero = logitsMax == 0.0f && logitsMin == 0.0f;
            StringBuilder first10 = new StringBuilder();
            int show = Math.min(10, len);
            for (int i = 0; i < show; i++) {
                if (i > 0) first10.append(", ");
                first10.append(String.format("%.4f", lastLogitsDiag.getFloat(i)));
            }
            log.info("  [DIAG] step={} logits: topId={} topVal={} min={} max={} mean={} hasNaN={} allZero={} shape={} first10=[{}]",
                    step, topId, topVal, logitsMin, logitsMax, logitsMean, hasNaN, allZero,
                    Arrays.toString(lastLogitsDiag.shape()), first10);
        } catch (Exception e) {
            log.warn("  [DIAG] step={} logits diagnostics failed: {}", step, e.getMessage());
        }
    }

    private static long percentile(List<Long> values, int percentile) {
        if (values.isEmpty()) return 0;
        List<Long> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int idx = (int) Math.ceil(percentile / 100.0 * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(idx, sorted.size() - 1)));
    }
}
