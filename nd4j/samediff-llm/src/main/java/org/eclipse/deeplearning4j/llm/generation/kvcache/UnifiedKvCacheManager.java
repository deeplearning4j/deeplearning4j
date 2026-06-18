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

package org.eclipse.deeplearning4j.llm.generation.kvcache;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheDequantize;
import org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheQuantize;
import org.nd4j.linalg.api.ops.impl.transforms.custom.KvScatter;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Unified KV cache manager for all decode strategies.
 *
 * <p>Consolidates Static, Quantized, TurboQuant, and Paged KV cache management
 * into a single class with one code path. KV scatter is expressed as a standard
 * graph op ({@link KvScatter}), captured into CUDA graph replay like any other op.
 * There is no bespoke C++ lifecycle — the DSP plan is a pure graph executor.</p>
 *
 * <h3>Strategies</h3>
 * <ul>
 *   <li>{@link KvCacheStrategy#STATIC} — Dense pre-allocated buffers, CUDA graph compatible</li>
 *   <li>{@link KvCacheStrategy#QUANTIZED} — INT8/FP8 quantized storage with per-step dequant</li>
 *   <li>{@link KvCacheStrategy#TURBOQUANT} — Two-stage vector quantization (MSE + QJL)</li>
 *   <li>{@link KvCacheStrategy#PAGED} — Block-based allocation via {@link PagedKVCache}</li>
 * </ul>
 *
 * <h3>In-Graph Scatter</h3>
 * <p>Call {@link #buildInGraphScatterOps} after prefill to add KvScatter nodes
 * directly into the SameDiff graph. These ops execute as part of the graph plan
 * and are captured into CUDA graph replay. When in-graph scatter is active,
 * {@link #scatterNewEntries} is a no-op — the graph handles everything.</p>
 *
 * @see KvCacheManager
 * @see KvScatter
 */
@Slf4j
public class UnifiedKvCacheManager implements KvCacheManager {

    private final KvCacheStrategy strategy;
    private final ModelIOConfig ioConfig;

    // ── Common state ──
    @Getter
    private Map<String, INDArray> staticKvBuffers;
    @Getter
    private long maxKvLen;
    @Getter
    private long cachePosition;
    private boolean initialized;

    // In-graph scatter: when enabled, scatterNewEntries() is a no-op
    private boolean inGraphScatterEnabled;
    @Getter
    private List<String> scatterOutputNames;

    // ── Quantized state (QUANTIZED strategy) ──
    @Getter
    private final QuantizedPagedKVCache.QuantFormat quantFormat;
    private final DataType originalDataType;
    private Map<String, INDArray> quantizedBuffers;
    private Map<String, INDArray> scaleBuffers;

    // ── TurboQuant state (TURBOQUANT strategy) ──
    private final int turboQuantBits;
    private Map<String, INDArray> rotationMatrices;
    private Map<String, INDArray> qjlMatrices;
    private Map<String, TurboQuantCodebook.Codebook> codebooks;
    private Map<String, INDArray> keyMseReconstruction;
    private Map<String, INDArray> keyQjlSigns;
    private Map<String, INDArray> keyResidualNorms;
    private Map<String, INDArray> valueMseIndices;
    private Map<String, INDArray> valueVecNorms;
    private List<String> turboKeyNames;
    private List<String> turboValueNames;

    // ── Paged state (PAGED strategy) ──
    private PagedKVCache[] layerCaches;
    private final int blockSize;
    private Map<String, Integer> nameToLayerIndex;
    private Map<String, Boolean> nameIsKey;
    private List<String> allPastNames;
    private int numLayers;

    // ── Shared dimension info ──
    private long numHeads;
    private long headDim;
    private DataType kvDataType;

    // ══════════════════════════════════════════════════════════════
    //  Constructors
    // ══════════════════════════════════════════════════════════════

    public UnifiedKvCacheManager(KvCacheStrategy strategy, ModelIOConfig ioConfig) {
        this(strategy, ioConfig, QuantizedPagedKVCache.QuantFormat.INT8,
                DataType.FLOAT, 3, PagedKVCache.DEFAULT_BLOCK_SIZE);
    }

    public UnifiedKvCacheManager(KvCacheStrategy strategy, ModelIOConfig ioConfig,
                                 QuantizedPagedKVCache.QuantFormat quantFormat,
                                 DataType originalDataType,
                                 int turboQuantBits,
                                 int blockSize) {
        this.strategy = strategy;
        this.ioConfig = ioConfig != null ? ioConfig : ModelIOConfig.builder().build();
        this.quantFormat = quantFormat;
        this.originalDataType = originalDataType;
        this.turboQuantBits = turboQuantBits;
        this.blockSize = blockSize;
        this.initialized = false;
        this.cachePosition = 0;
        this.maxKvLen = -1;
        this.inGraphScatterEnabled = false;
    }

    public UnifiedKvCacheManager() {
        this(KvCacheStrategy.STATIC, ModelIOConfig.builder().build());
    }

    // ══════════════════════════════════════════════════════════════
    //  Private KV shape inference (replaces inferEmptyKvBuffer)
    // ══════════════════════════════════════════════════════════════

    /**
     * Create an empty KV cache tensor [batchSize, numHeads, 0, headDim] by inferring
     * head count and dimension from the decoder graph's variable shapes.
     */
    private static INDArray inferEmptyKvBuffer(SameDiff decoder, String inputName,
                                               long batchSize, long hiddenSize) {
        long numH = -1;
        long hDim = -1;
        DataType kvType = DataType.FLOAT;

        SDVariable inputVar = decoder.getVariable(inputName);
        if (inputVar != null && inputVar.getShape() != null && inputVar.getShape().length >= 4) {
            long[] shape = inputVar.getShape();
            if (inputVar.dataType() != null) kvType = inputVar.dataType();
            if (shape[1] > 0) numH = shape[1];
            if (shape[3] > 0) hDim = shape[3];
        }

        if (numH <= 0 || hDim <= 0) {
            ModelIOConfig defaultConfig = ModelIOConfig.builder().build();
            String presentName = defaultConfig.inputToPresentName(inputName);
            SDVariable presentVar = decoder.getVariable(presentName);
            if (presentVar != null && presentVar.getShape() != null && presentVar.getShape().length >= 4) {
                long[] shape = presentVar.getShape();
                if (numH <= 0 && shape[1] > 0) numH = shape[1];
                if (hDim <= 0 && shape[3] > 0) hDim = shape[3];
            }
        }

        if (hDim <= 0 && numH > 0 && hiddenSize > 0) hDim = Math.max(1, hiddenSize / numH);
        if (numH <= 0 && hDim > 0 && hiddenSize > 0) numH = Math.max(1, hiddenSize / hDim);
        if (hDim <= 0) hDim = 64;
        if (numH <= 0) numH = Math.max(1, hiddenSize / hDim);

        return Nd4j.zeros(kvType, batchSize, numH, 0, hDim);
    }

    // ══════════════════════════════════════════════════════════════
    //  KvCacheManager interface
    // ══════════════════════════════════════════════════════════════

    @Override
    public KvCacheStrategy getStrategy() {
        return strategy;
    }

    @Override
    public void initializeFromPrefill(Map<String, INDArray> prefillOutputs,
                                      ModelIOConfig.KVCacheNames kvNames,
                                      int maxNewTokens, long prefillSeqLen) {
        this.maxKvLen = prefillSeqLen + maxNewTokens;

        log.info("  UnifiedKvCacheManager[{}]: prefillLen={}, maxKvLen={}, {} key + {} value tensors",
                strategy, prefillSeqLen, maxKvLen, kvNames.keyNames.size(), kvNames.valueNames.size());

        switch (strategy) {
            case STATIC:
                initStaticFromPrefill(prefillOutputs, kvNames, prefillSeqLen);
                break;
            case QUANTIZED:
                initQuantizedFromPrefill(prefillOutputs, kvNames, prefillSeqLen);
                break;
            case TURBOQUANT:
                initTurboQuantFromPrefill(prefillOutputs, kvNames, prefillSeqLen);
                break;
            case PAGED:
                initPagedFromPrefill(prefillOutputs, kvNames, prefillSeqLen);
                break;
            default:
                throw new IllegalStateException("Unknown KV cache strategy: " + strategy);
        }

        Nd4j.getExecutioner().commit();
        this.cachePosition = prefillSeqLen;
        this.initialized = true;
    }

    @Override
    public void initializeEmpty(SameDiff decoder, List<String> decoderInputNames,
                                int maxKvLen, long hiddenSize) {
        this.maxKvLen = maxKvLen;

        List<String> pastInputNames = new ArrayList<>();
        for (String inputName : decoderInputNames) {
            if (ioConfig.isKvCacheInput(inputName)) {
                pastInputNames.add(inputName);
            }
        }

        switch (strategy) {
            case STATIC:
                initStaticEmpty(decoder, pastInputNames, hiddenSize);
                break;
            case QUANTIZED:
                initQuantizedEmpty(decoder, pastInputNames, hiddenSize);
                break;
            case TURBOQUANT:
                initTurboQuantEmpty(decoder, pastInputNames, hiddenSize);
                break;
            case PAGED:
                initPagedEmpty(decoder, pastInputNames, hiddenSize);
                break;
            default:
                throw new IllegalStateException("Unknown KV cache strategy: " + strategy);
        }

        this.cachePosition = 0;
        this.initialized = true;
        log.info("  UnifiedKvCacheManager[{}]: initialized empty with maxKvLen={}", strategy, maxKvLen);
    }

    @Override
    public void prepareInputs(Map<String, INDArray> decoderInputMap, SameDiff decoder,
                              long hiddenSize, boolean dspActive) {
        if (!initialized) return;

        switch (strategy) {
            case STATIC:
            case PAGED:
                prepareStaticInputs(decoderInputMap, decoder, hiddenSize, dspActive);
                break;
            case QUANTIZED:
                prepareQuantizedInputs(decoderInputMap, decoder, hiddenSize, dspActive);
                break;
            case TURBOQUANT:
                prepareTurboQuantInputs(decoderInputMap, decoder, hiddenSize, dspActive);
                break;
        }
    }

    @Override
    public void scatterNewEntries(Map<String, INDArray> decoderOutputs,
                                  ModelIOConfig.KVCacheNames kvNames) {
        if (inGraphScatterEnabled) return;

        switch (strategy) {
            case STATIC:
                scatterStaticKvEntries(decoderOutputs, kvNames);
                break;
            case QUANTIZED:
                scatterQuantizedEntries(decoderOutputs, kvNames, 1);
                break;
            case TURBOQUANT:
                scatterTurboQuantEntries(decoderOutputs, kvNames, 1);
                break;
            case PAGED:
                scatterPagedEntries(decoderOutputs, kvNames, 1);
                break;
        }

        cachePosition++;
    }

    @Override
    public void scatterMultipleEntries(Map<String, INDArray> decoderOutputs,
                                       ModelIOConfig.KVCacheNames kvNames,
                                       int numEntries) {
        if (numEntries <= 0) return;
        if (inGraphScatterEnabled) return;

        switch (strategy) {
            case STATIC:
                scatterStaticMultipleKvEntries(decoderOutputs, kvNames, numEntries);
                break;
            case QUANTIZED:
                scatterQuantizedEntries(decoderOutputs, kvNames, numEntries);
                break;
            case TURBOQUANT:
                scatterTurboQuantEntries(decoderOutputs, kvNames, numEntries);
                break;
            case PAGED:
                scatterPagedEntries(decoderOutputs, kvNames, numEntries);
                break;
        }

        cachePosition += numEntries;
    }

    @Override
    public void setCachePosition(long position) {
        this.cachePosition = position;
    }

    @Override
    public boolean isInitialized() {
        return initialized;
    }

    @Override
    public boolean supportsCudaGraphReplay() {
        return strategy == KvCacheStrategy.STATIC || strategy == KvCacheStrategy.PAGED;
    }

    @Override
    public Map<String, INDArray> getStaticKvBuffers() {
        return staticKvBuffers;
    }

    // ══════════════════════════════════════════════════════════════
    //  In-Graph Scatter — the unified code path
    // ══════════════════════════════════════════════════════════════

    /**
     * Add KvScatter ops to the SameDiff graph for in-graph KV cache updates.
     *
     * <p>After calling this, the graph contains scatter ops that write present KV
     * outputs into the static KV buffers at the position specified by the
     * {@code kv_cache_position} placeholder. These ops are captured into CUDA
     * graph replay like any other graph node.</p>
     *
     * <p>Once enabled, {@link #scatterNewEntries} becomes a no-op — the graph
     * handles all KV updates. The caller must:</p>
     * <ol>
     *   <li>Add {@code kv_cache_position} (INT64, shape [1]) to the input map each step</li>
     *   <li>Request {@link #getScatterOutputNames()} as graph outputs to ensure scatter ops execute</li>
     *   <li>No longer call {@link #scatterNewEntries} or {@link #scatterMultipleEntries}</li>
     * </ol>
     *
     * @param sd      the SameDiff graph to augment
     * @param kvNames present KV output names from the decoder
     * @return list of scatter output variable names (must be requested as graph outputs)
     */
    public List<String> buildInGraphScatterOps(SameDiff sd, ModelIOConfig.KVCacheNames kvNames) {
        if (staticKvBuffers == null || staticKvBuffers.isEmpty()) {
            throw new IllegalStateException(
                    "Static KV buffers must be initialized before building in-graph scatter ops");
        }

        this.scatterOutputNames = new ArrayList<>();

        List<String> allPresentNames = new ArrayList<>();
        allPresentNames.addAll(kvNames.keyNames);
        allPresentNames.addAll(kvNames.valueNames);

        for (String presentName : allPresentNames) {
            String pastName = ioConfig.presentToInputName(presentName);

            if (!sd.hasVariable(presentName) || !sd.hasVariable(pastName)) {
                log.warn("buildInGraphScatterOps: skipping {} — present or past variable not found in graph",
                        presentName);
                continue;
            }

            SDVariable present = sd.getVariable(presentName);
            SDVariable past = sd.getVariable(pastName);

            String scatterVarName = "kv_scatter__" + pastName.replace(".", "_");

            KvScatter scatterOp = new KvScatter(sd, past, present, cachePosition, 1);
            SDVariable scatterOut = sd.updateVariableNameAndReference(scatterOp.outputVariable(), scatterVarName);

            scatterOutputNames.add(scatterVarName);
        }

        this.inGraphScatterEnabled = true;
        log.info("  UnifiedKvCacheManager: built {} in-graph scatter ops", scatterOutputNames.size());
        return scatterOutputNames;
    }

    /**
     * Whether in-graph scatter is active. When true, scatterNewEntries is a no-op
     * and the caller must request scatter output names as graph outputs.
     */
    public boolean isInGraphScatterEnabled() {
        return inGraphScatterEnabled;
    }

    // ══════════════════════════════════════════════════════════════
    //  STATIC strategy implementation
    // ══════════════════════════════════════════════════════════════

    private void initStaticFromPrefill(Map<String, INDArray> prefillOutputs,
                                       ModelIOConfig.KVCacheNames kvNames,
                                       long prefillSeqLen) {
        this.staticKvBuffers = new HashMap<>();
        List<String> allPresentNames = new ArrayList<>(kvNames.keyNames);
        allPresentNames.addAll(kvNames.valueNames);

        Nd4j.getExecutioner().commit();

        for (String presentName : allPresentNames) {
            INDArray prefillKv = prefillOutputs.get(presentName);
            if (prefillKv == null) continue;

            long[] shape = prefillKv.shape();
            long batch = shape[0];
            long heads = shape[1];
            long prefillLen = shape[2];
            long dim = shape[3];
            long padLen = maxKvLen - prefillLen;

            // Build static buffer by concatenating prefill data with zero padding
            // along the sequence dimension (axis 2). Nd4j.concat runs as a native op
            // on device — no manual host/device sync needed.
            INDArray staticBuf;
            if (padLen > 0) {
                INDArray padding = Nd4j.zeros(prefillKv.dataType(), batch, heads, padLen, dim);
                staticBuf = Nd4j.concat(2, prefillKv, padding);
                padding.close();
            } else {
                staticBuf = prefillKv.dup();
            }

            String pastInputName = ioConfig.presentToInputName(presentName);
            staticBuf.setCloseable(false);
            staticKvBuffers.put(pastInputName, staticBuf);

            log.info("  Static KV buffer '{}': [{},{},{},{}] (prefill={} padded to {})",
                    pastInputName, batch, heads, maxKvLen, dim, prefillLen, maxKvLen);
        }
    }

    private void initStaticEmpty(SameDiff decoder, List<String> pastInputNames, long hiddenSize) {
        this.staticKvBuffers = new HashMap<>();
        for (String pastInputName : pastInputNames) {
            INDArray empty = inferEmptyKvBuffer(decoder, pastInputName, 1, hiddenSize);
            long numH = empty.size(1);
            long hDim = empty.size(3);
            DataType kvType = empty.dataType();
            empty.close();

            INDArray buf = Nd4j.zeros(kvType, 1, numH, maxKvLen, hDim);
            buf.setCloseable(false);
            staticKvBuffers.put(pastInputName, buf);
        }
    }

    private void scatterStaticKvEntries(Map<String, INDArray> decoderOutputs,
                                          ModelIOConfig.KVCacheNames kvNames) {
        List<INDArray> staticList = new ArrayList<>();
        List<INDArray> presentList = new ArrayList<>();

        List<String> allPresentNames = new ArrayList<>(kvNames.keyNames);
        allPresentNames.addAll(kvNames.valueNames);

        int skippedClosed = 0;
        for (String presentName : allPresentNames) {
            INDArray presentKv = decoderOutputs.get(presentName);
            if (presentKv == null) continue;

            String pastInputName = ioConfig.presentToInputName(presentName);
            INDArray staticBuf = staticKvBuffers.get(pastInputName);
            if (staticBuf != null) {
                if (staticBuf.wasClosed() || staticBuf.data() == null) { skippedClosed++; continue; }
                if (presentKv.wasClosed() || presentKv.data() == null) { skippedClosed++; continue; }
                staticList.add(staticBuf);
                presentList.add(presentKv);
            }
        }

        if (skippedClosed > 0) {
            log.warn("scatterStaticKvEntries: skipped {} closed arrays at cachePos={}", skippedClosed, cachePosition);
        }
        if (staticList.isEmpty()) return;

        Nd4j.exec(new KvScatter(
                staticList.toArray(new INDArray[0]),
                presentList.toArray(new INDArray[0]),
                cachePosition));
    }

    private void scatterStaticMultipleKvEntries(Map<String, INDArray> decoderOutputs,
                                                 ModelIOConfig.KVCacheNames kvNames,
                                                 int numPositions) {
        if (numPositions <= 0) return;

        List<String> allPresentNames = new ArrayList<>(kvNames.keyNames);
        allPresentNames.addAll(kvNames.valueNames);

        for (String presentName : allPresentNames) {
            INDArray presentKv = decoderOutputs.get(presentName);
            if (presentKv == null) continue;

            String pastInputName = ioConfig.presentToInputName(presentName);
            INDArray staticBuf = staticKvBuffers.get(pastInputName);
            if (staticBuf == null) continue;

            INDArray sourceSlice = presentKv.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(maxKvLen, maxKvLen + numPositions),
                    NDArrayIndex.all());

            INDArray destSlice = staticBuf.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(cachePosition, cachePosition + numPositions),
                    NDArrayIndex.all());
            destSlice.assign(sourceSlice);
        }
    }

    private void prepareStaticInputs(Map<String, INDArray> decoderInputMap, SameDiff decoder,
                                     long hiddenSize, boolean dspActive) {
        if (staticKvBuffers == null) return;

        for (String inputName : decoder.inputs()) {
            if (!ioConfig.isKvCacheInput(inputName)) continue;

            INDArray staticBuf = staticKvBuffers.get(inputName);
            if (staticBuf != null) {
                if (dspActive) {
                    decoderInputMap.put(inputName, staticBuf);
                } else {
                    if (cachePosition > 0 && cachePosition < staticBuf.size(2)) {
                        INDArray view = staticBuf.get(
                                NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(0, cachePosition), NDArrayIndex.all());
                        decoderInputMap.put(inputName, view);
                    } else if (cachePosition >= staticBuf.size(2)) {
                        decoderInputMap.put(inputName, staticBuf);
                    } else {
                        decoderInputMap.put(inputName,
                                inferEmptyKvBuffer(decoder, inputName, 1, hiddenSize));
                    }
                }
            } else {
                decoderInputMap.put(inputName,
                        inferEmptyKvBuffer(decoder, inputName, 1, hiddenSize));
            }
        }
    }

    // ══════════════════════════════════════════════════════════════
    //  QUANTIZED strategy implementation
    // ══════════════════════════════════════════════════════════════

    private void initQuantizedFromPrefill(Map<String, INDArray> prefillOutputs,
                                          ModelIOConfig.KVCacheNames kvNames,
                                          long prefillSeqLen) {
        this.quantizedBuffers = new HashMap<>();
        this.scaleBuffers = new HashMap<>();
        this.staticKvBuffers = new HashMap<>();

        List<String> allNames = new ArrayList<>(kvNames.keyNames);
        allNames.addAll(kvNames.valueNames);

        for (String name : allNames) {
            String presentName = ioConfig.inputToPresentName(name);
            INDArray presentKv = findPresentOutput(prefillOutputs, presentName, name);
            if (presentKv == null) continue;

            this.numHeads = presentKv.size(1);
            this.headDim = presentKv.size(3);
            this.kvDataType = presentKv.dataType();

            quantizedBuffers.put(name, Nd4j.zeros(DataType.INT8, 1, numHeads, maxKvLen, headDim));
            scaleBuffers.put(name, Nd4j.ones(DataType.FLOAT, 1, numHeads, maxKvLen));
            staticKvBuffers.put(name, Nd4j.zeros(kvDataType, 1, numHeads, maxKvLen, headDim));

            quantizeAndStore(presentKv, name, 0, prefillSeqLen);
        }
    }

    private void initQuantizedEmpty(SameDiff decoder, List<String> pastInputNames, long hiddenSize) {
        this.quantizedBuffers = new HashMap<>();
        this.scaleBuffers = new HashMap<>();
        this.staticKvBuffers = new HashMap<>();

        for (String pastInputName : pastInputNames) {
            INDArray empty = inferEmptyKvBuffer(decoder, pastInputName, 1, hiddenSize);
            long numH = empty.size(1);
            long hDim = empty.size(3);
            DataType kvType = empty.dataType();
            empty.close();

            this.numHeads = numH;
            this.headDim = hDim;
            this.kvDataType = kvType;

            quantizedBuffers.put(pastInputName, Nd4j.zeros(DataType.INT8, 1, numH, maxKvLen, hDim));
            scaleBuffers.put(pastInputName, Nd4j.ones(DataType.FLOAT, 1, numH, maxKvLen));
            staticKvBuffers.put(pastInputName, Nd4j.zeros(kvType, 1, numH, maxKvLen, hDim));
        }
    }

    private void prepareQuantizedInputs(Map<String, INDArray> decoderInputMap, SameDiff decoder,
                                         long hiddenSize, boolean dspActive) {
        if (quantizedBuffers == null) return;

        for (String inputName : decoder.inputs()) {
            if (!ioConfig.isKvCacheInput(inputName)) continue;

            INDArray deqBuf = staticKvBuffers.get(inputName);
            if (deqBuf != null && cachePosition > 0) {
                dequantizeRegion(inputName, 0, cachePosition);
                if (dspActive) {
                    decoderInputMap.put(inputName, deqBuf);
                } else {
                    INDArray view = deqBuf.get(
                            NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.interval(0, cachePosition), NDArrayIndex.all());
                    decoderInputMap.put(inputName, view);
                }
            } else {
                decoderInputMap.put(inputName,
                        inferEmptyKvBuffer(decoder, inputName, 1, hiddenSize));
            }
        }
    }

    private void scatterQuantizedEntries(Map<String, INDArray> decoderOutputs,
                                          ModelIOConfig.KVCacheNames kvNames,
                                          int numEntries) {
        List<String> allPastNames = new ArrayList<>(kvNames.keyNames);
        allPastNames.addAll(kvNames.valueNames);

        for (String pastName : allPastNames) {
            String presentName = ioConfig.inputToPresentName(pastName);
            INDArray presentKv = findPresentOutput(decoderOutputs, presentName, pastName);
            if (presentKv == null) continue;

            long totalSeqLen = presentKv.size(2);
            long sourceStart = totalSeqLen - numEntries;
            if (sourceStart < 0) sourceStart = 0;
            if (totalSeqLen > maxKvLen) sourceStart = maxKvLen;

            INDArray newEntries = presentKv.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(sourceStart, sourceStart + numEntries),
                    NDArrayIndex.all());

            quantizeAndStore(newEntries, pastName, cachePosition, numEntries);
        }
    }

    private void quantizeAndStore(INDArray kvData, String bufferName,
                                   long startPos, long numTokens) {
        INDArray quantBuf = quantizedBuffers.get(bufferName);
        INDArray scaleBuf = scaleBuffers.get(bufferName);
        if (quantBuf == null || scaleBuf == null) return;

        long heads = kvData.size(1);
        long hDim = kvData.size(3);

        INDArray kvReshaped = kvData.permute(0, 2, 1, 3).reshape(numTokens * heads, hDim);
        KVCacheQuantize quantOp = new KVCacheQuantize(kvReshaped, quantFormat.getNativeCode());
        INDArray[] quantResult = Nd4j.exec(quantOp);
        INDArray quantized = quantResult[0];
        INDArray scales = quantResult[1];

        INDArray quantReshaped = quantized.reshape(1, numTokens, heads, hDim).permute(0, 2, 1, 3);
        INDArray scalesReshaped = scales.reshape(1, numTokens, heads).permute(0, 2, 1);

        quantBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(startPos, startPos + numTokens), NDArrayIndex.all())
                .assign(quantReshaped);
        scaleBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(startPos, startPos + numTokens))
                .assign(scalesReshaped);
    }

    private void dequantizeRegion(String bufferName, long startPos, long endPos) {
        INDArray quantBuf = quantizedBuffers.get(bufferName);
        INDArray scaleBuf = scaleBuffers.get(bufferName);
        INDArray deqBuf = staticKvBuffers.get(bufferName);
        if (quantBuf == null || scaleBuf == null || deqBuf == null) return;

        long numTokens = endPos - startPos;
        if (numTokens <= 0) return;

        INDArray quantRegion = quantBuf.get(
                NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(startPos, endPos), NDArrayIndex.all());
        INDArray scaleRegion = scaleBuf.get(
                NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(startPos, endPos));

        long heads = quantRegion.size(1);
        long hDim = quantRegion.size(3);

        INDArray quantFlat = quantRegion.permute(0, 2, 1, 3).reshape(numTokens * heads, hDim);
        INDArray scaleFlat = scaleRegion.permute(0, 2, 1).reshape(numTokens * heads);

        KVCacheDequantize deqOp = new KVCacheDequantize(quantFlat, scaleFlat, quantFormat.getNativeCode());
        INDArray[] deqResult = Nd4j.exec(deqOp);
        INDArray dequantized = deqResult[0];

        INDArray deqReshaped = dequantized.reshape(1, numTokens, heads, hDim).permute(0, 2, 1, 3);

        INDArray deqDest = deqBuf.get(
                NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(startPos, endPos), NDArrayIndex.all());
        if (deqBuf.dataType() != dequantized.dataType()) {
            deqDest.assign(deqReshaped.castTo(deqBuf.dataType()));
        } else {
            deqDest.assign(deqReshaped);
        }
    }

    // ══════════════════════════════════════════════════════════════
    //  TURBOQUANT strategy implementation
    // ══════════════════════════════════════════════════════════════

    private void initTurboQuantFromPrefill(Map<String, INDArray> prefillOutputs,
                                            ModelIOConfig.KVCacheNames kvNames,
                                            long prefillSeqLen) {
        this.turboKeyNames = new ArrayList<>(kvNames.keyNames);
        this.turboValueNames = new ArrayList<>(kvNames.valueNames);
        this.rotationMatrices = new HashMap<>();
        this.qjlMatrices = new HashMap<>();
        this.codebooks = new HashMap<>();
        this.keyMseReconstruction = new HashMap<>();
        this.keyQjlSigns = new HashMap<>();
        this.keyResidualNorms = new HashMap<>();
        this.valueMseIndices = new HashMap<>();
        this.valueVecNorms = new HashMap<>();
        this.staticKvBuffers = new HashMap<>();

        Random rng = new Random(42);

        for (String name : turboKeyNames) {
            String presentName = ioConfig.inputToPresentName(name);
            INDArray presentKv = findPresentOutput(prefillOutputs, presentName, name);
            if (presentKv == null) continue;

            this.numHeads = presentKv.size(1);
            this.headDim = presentKv.size(3);
            int d = (int) headDim;

            rotationMatrices.put(name, generateOrthogonalMatrix(d, rng));
            qjlMatrices.put(name, generateGaussianMatrix(d, d, rng));
            codebooks.put(name, TurboQuantCodebook.getCodebook(d, Math.max(turboQuantBits - 1, 1)));

            keyMseReconstruction.put(name, Nd4j.zeros(DataType.HALF, 1, numHeads, maxKvLen, headDim));
            keyQjlSigns.put(name, Nd4j.zeros(DataType.INT8, 1, numHeads, maxKvLen, headDim));
            keyResidualNorms.put(name, Nd4j.zeros(DataType.HALF, 1, numHeads, maxKvLen));

            compressKeys(presentKv, name, 0, prefillSeqLen);
            staticKvBuffers.put(name, keyMseReconstruction.get(name));
        }

        for (String name : turboValueNames) {
            String presentName = ioConfig.inputToPresentName(name);
            INDArray presentKv = findPresentOutput(prefillOutputs, presentName, name);
            if (presentKv == null) continue;

            this.numHeads = presentKv.size(1);
            this.headDim = presentKv.size(3);
            this.kvDataType = presentKv.dataType();
            int d = (int) headDim;

            if (!rotationMatrices.containsKey(name)) {
                rotationMatrices.put(name, generateOrthogonalMatrix(d, rng));
                codebooks.put(name, TurboQuantCodebook.getCodebook(d, turboQuantBits));
            }

            valueMseIndices.put(name, Nd4j.zeros(DataType.INT8, 1, numHeads, maxKvLen, headDim));
            valueVecNorms.put(name, Nd4j.zeros(DataType.HALF, 1, numHeads, maxKvLen));

            INDArray deqBuf = Nd4j.zeros(kvDataType, 1, numHeads, maxKvLen, headDim);
            staticKvBuffers.put(name, deqBuf);

            compressValues(presentKv, name, 0, prefillSeqLen);
            dequantizeValues(name, 0, prefillSeqLen);
        }
    }

    private void initTurboQuantEmpty(SameDiff decoder, List<String> pastInputNames, long hiddenSize) {
        this.turboKeyNames = new ArrayList<>();
        this.turboValueNames = new ArrayList<>();
        this.rotationMatrices = new HashMap<>();
        this.qjlMatrices = new HashMap<>();
        this.codebooks = new HashMap<>();
        this.keyMseReconstruction = new HashMap<>();
        this.keyQjlSigns = new HashMap<>();
        this.keyResidualNorms = new HashMap<>();
        this.valueMseIndices = new HashMap<>();
        this.valueVecNorms = new HashMap<>();
        this.staticKvBuffers = new HashMap<>();

        Random rng = new Random(42);

        for (String inputName : pastInputNames) {
            INDArray empty = inferEmptyKvBuffer(decoder, inputName, 1, hiddenSize);
            long numH = empty.size(1);
            long hDim = empty.size(3);
            DataType kvType = empty.dataType();
            empty.close();

            this.numHeads = numH;
            this.headDim = hDim;
            int d = (int) hDim;
            boolean isKey = inputName.contains("key");

            if (isKey) {
                turboKeyNames.add(inputName);
                rotationMatrices.put(inputName, generateOrthogonalMatrix(d, rng));
                qjlMatrices.put(inputName, generateGaussianMatrix(d, d, rng));
                codebooks.put(inputName, TurboQuantCodebook.getCodebook(d, Math.max(turboQuantBits - 1, 1)));
                keyMseReconstruction.put(inputName, Nd4j.zeros(DataType.HALF, 1, numH, maxKvLen, hDim));
                keyQjlSigns.put(inputName, Nd4j.zeros(DataType.INT8, 1, numH, maxKvLen, hDim));
                keyResidualNorms.put(inputName, Nd4j.zeros(DataType.HALF, 1, numH, maxKvLen));
                staticKvBuffers.put(inputName, keyMseReconstruction.get(inputName));
            } else {
                turboValueNames.add(inputName);
                rotationMatrices.put(inputName, generateOrthogonalMatrix(d, rng));
                codebooks.put(inputName, TurboQuantCodebook.getCodebook(d, turboQuantBits));
                valueMseIndices.put(inputName, Nd4j.zeros(DataType.INT8, 1, numH, maxKvLen, hDim));
                valueVecNorms.put(inputName, Nd4j.zeros(DataType.HALF, 1, numH, maxKvLen));
                staticKvBuffers.put(inputName, Nd4j.zeros(kvType, 1, numH, maxKvLen, hDim));
            }
        }
    }

    private void prepareTurboQuantInputs(Map<String, INDArray> decoderInputMap, SameDiff decoder,
                                          long hiddenSize, boolean dspActive) {
        if (!initialized) return;

        for (String inputName : decoder.inputs()) {
            if (!ioConfig.isKvCacheInput(inputName)) continue;

            if (staticKvBuffers.containsKey(inputName) && cachePosition > 0) {
                boolean isValue = valueMseIndices != null && valueMseIndices.containsKey(inputName);
                if (isValue) {
                    dequantizeValues(inputName, 0, cachePosition);
                }
                decoderInputMap.put(inputName, staticKvBuffers.get(inputName));
            } else {
                decoderInputMap.put(inputName,
                        inferEmptyKvBuffer(decoder, inputName, 1, hiddenSize));
            }
        }
    }

    private void scatterTurboQuantEntries(Map<String, INDArray> decoderOutputs,
                                           ModelIOConfig.KVCacheNames kvNames,
                                           int numEntries) {
        if (turboKeyNames != null) {
            for (String keyName : kvNames.keyNames) {
                String presentName = ioConfig.inputToPresentName(keyName);
                INDArray presentKv = findPresentOutput(decoderOutputs, presentName, keyName);
                if (presentKv == null) continue;

                long totalSeqLen = presentKv.size(2);
                long sourceStart = Math.max(0, totalSeqLen - numEntries);
                if (totalSeqLen > maxKvLen) sourceStart = maxKvLen;

                INDArray newEntries = presentKv.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(sourceStart, sourceStart + numEntries),
                        NDArrayIndex.all());
                compressKeys(newEntries, keyName, cachePosition, numEntries);
            }
        }

        if (turboValueNames != null) {
            for (String valueName : kvNames.valueNames) {
                String presentName = ioConfig.inputToPresentName(valueName);
                INDArray presentKv = findPresentOutput(decoderOutputs, presentName, valueName);
                if (presentKv == null) continue;

                long totalSeqLen = presentKv.size(2);
                long sourceStart = Math.max(0, totalSeqLen - numEntries);
                if (totalSeqLen > maxKvLen) sourceStart = maxKvLen;

                INDArray newEntries = presentKv.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(sourceStart, sourceStart + numEntries),
                        NDArrayIndex.all());
                compressValues(newEntries, valueName, cachePosition, numEntries);
                dequantizeValues(valueName, cachePosition, cachePosition + numEntries);
            }
        }
    }

    // ── TurboQuant compression/decompression (MSE + QJL) ──

    private void compressKeys(INDArray kvData, String bufferName, long startPos, long numTokens) {
        INDArray rotation = rotationMatrices.get(bufferName);
        INDArray qjl = qjlMatrices.get(bufferName);
        TurboQuantCodebook.Codebook codebook = codebooks.get(bufferName);
        if (rotation == null || qjl == null || codebook == null) return;

        INDArray kmseBuf = keyMseReconstruction.get(bufferName);
        INDArray signsBuf = keyQjlSigns.get(bufferName);
        INDArray normsBuf = keyResidualNorms.get(bufferName);
        if (kmseBuf == null || signsBuf == null || normsBuf == null) return;

        int d = (int) headDim;
        int H = (int) numHeads;

        for (int h = 0; h < H; h++) {
            for (long t = 0; t < numTokens; t++) {
                INDArray keyVec = kvData.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                        NDArrayIndex.point(t), NDArrayIndex.all());
                INDArray keyFloat = keyVec.castTo(DataType.FLOAT);

                double norm = keyFloat.norm2Number().doubleValue();
                long pos = startPos + t;

                if (norm < 1e-10) {
                    kmseBuf.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                            NDArrayIndex.point(pos), NDArrayIndex.all()).assign(0);
                    signsBuf.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                            NDArrayIndex.point(pos), NDArrayIndex.all()).assign(0);
                    normsBuf.putScalar(new long[]{0, h, pos}, 0);
                    continue;
                }

                INDArray normalized = keyFloat.div(norm).reshape(1, d);
                INDArray rotated = normalized.mmul(rotation.transpose());

                float[] rotatedData = rotated.dup().data().asFloat();
                float[] reconstructed = new float[d];
                for (int i = 0; i < d; i++) {
                    int idx = TurboQuantCodebook.quantize(rotatedData[i], codebook);
                    reconstructed[i] = TurboQuantCodebook.dequantize(idx, codebook);
                }

                INDArray reconArr = Nd4j.createFromArray(reconstructed).reshape(1, d);
                INDArray unrotated = reconArr.mmul(rotation);
                INDArray kMse = unrotated.muli(norm).reshape(d);

                INDArray residual = keyFloat.sub(kMse);
                double residualNorm = residual.norm2Number().doubleValue();

                INDArray projected = residual.reshape(1, d).mmul(qjl.transpose());
                float[] projData = projected.dup().data().asFloat();
                byte[] signs = new byte[d];
                for (int i = 0; i < d; i++) {
                    signs[i] = (byte) (projData[i] >= 0 ? 1 : -1);
                }

                kmseBuf.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                        NDArrayIndex.point(pos), NDArrayIndex.all()).assign(kMse.castTo(DataType.HALF));
                signsBuf.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                        NDArrayIndex.point(pos), NDArrayIndex.all()).assign(
                        Nd4j.createFromArray(signs).reshape(d));
                normsBuf.putScalar(new long[]{0, h, pos}, (float) residualNorm);

                reconArr.close();
                unrotated.close();
            }
        }
    }

    private void compressValues(INDArray kvData, String bufferName, long startPos, long numTokens) {
        INDArray rotation = rotationMatrices.get(bufferName);
        TurboQuantCodebook.Codebook codebook = codebooks.get(bufferName);
        if (rotation == null || codebook == null) return;

        INDArray indicesBuf = valueMseIndices.get(bufferName);
        INDArray normsBuf = valueVecNorms.get(bufferName);
        if (indicesBuf == null || normsBuf == null) return;

        int d = (int) headDim;
        int H = (int) numHeads;

        for (int h = 0; h < H; h++) {
            for (long t = 0; t < numTokens; t++) {
                INDArray valVec = kvData.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                        NDArrayIndex.point(t), NDArrayIndex.all());
                INDArray valFloat = valVec.castTo(DataType.FLOAT);

                double norm = valFloat.norm2Number().doubleValue();
                long pos = startPos + t;

                if (norm < 1e-10) {
                    indicesBuf.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                            NDArrayIndex.point(pos), NDArrayIndex.all()).assign(0);
                    normsBuf.putScalar(new long[]{0, h, pos}, 0);
                    continue;
                }

                INDArray normalized = valFloat.div(norm).reshape(1, d);
                INDArray rotated = normalized.mmul(rotation.transpose());

                float[] rotatedData = rotated.dup().data().asFloat();
                byte[] indices = new byte[d];
                for (int i = 0; i < d; i++) {
                    indices[i] = (byte) TurboQuantCodebook.quantize(rotatedData[i], codebook);
                }

                indicesBuf.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                        NDArrayIndex.point(pos), NDArrayIndex.all()).assign(
                        Nd4j.createFromArray(indices).reshape(d));
                normsBuf.putScalar(new long[]{0, h, pos}, (float) norm);
            }
        }
    }

    private void dequantizeValues(String bufferName, long startPos, long endPos) {
        INDArray rotation = rotationMatrices.get(bufferName);
        TurboQuantCodebook.Codebook codebook = codebooks.get(bufferName);
        INDArray indicesBuf = valueMseIndices.get(bufferName);
        INDArray normsBuf = valueVecNorms.get(bufferName);
        INDArray deqBuf = staticKvBuffers.get(bufferName);
        if (rotation == null || codebook == null || indicesBuf == null
                || normsBuf == null || deqBuf == null) return;

        int d = (int) headDim;
        int H = (int) numHeads;

        for (int h = 0; h < H; h++) {
            // Bulk-fetch all norms for this head/range in one transfer instead of per-token getFloat
            INDArray normsSlice = normsBuf.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                    NDArrayIndex.interval(startPos, endPos));
            float[] normsArr = normsSlice.dup().data().asFloat();

            for (long t = startPos; t < endPos; t++) {
                float norm = normsArr[(int)(t - startPos)];

                if (Math.abs(norm) < 1e-10) {
                    deqBuf.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                            NDArrayIndex.point(t), NDArrayIndex.all()).assign(0);
                    continue;
                }

                // Bulk-fetch index vector for this token in one transfer instead of d scalar getInt calls
                INDArray idxVec = indicesBuf.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                        NDArrayIndex.point(t), NDArrayIndex.all());
                int[] idxData = idxVec.dup().data().asInt();

                float[] dequantized = new float[d];
                for (int i = 0; i < d; i++) {
                    int idx = idxData[i] & 0xFF;
                    dequantized[i] = TurboQuantCodebook.dequantize(idx, codebook);
                }

                INDArray deqArr = Nd4j.createFromArray(dequantized).reshape(1, d);
                INDArray unrotated = deqArr.mmul(rotation).muli(norm).reshape(d);

                deqBuf.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                        NDArrayIndex.point(t), NDArrayIndex.all())
                        .assign(unrotated.castTo(deqBuf.dataType()));

                deqArr.close();
                unrotated.close();
            }
        }
    }

    // ══════════════════════════════════════════════════════════════
    //  PAGED strategy implementation
    // ══════════════════════════════════════════════════════════════

    private void initPagedFromPrefill(Map<String, INDArray> prefillOutputs,
                                       ModelIOConfig.KVCacheNames kvNames,
                                       long prefillSeqLen) {
        this.numLayers = kvNames.keyNames.size();
        this.nameToLayerIndex = new HashMap<>();
        this.nameIsKey = new HashMap<>();
        this.allPastNames = new ArrayList<>();
        this.staticKvBuffers = new HashMap<>();

        for (int i = 0; i < numLayers; i++) {
            String keyPastName = ioConfig.presentToInputName(kvNames.keyNames.get(i));
            String valPastName = ioConfig.presentToInputName(kvNames.valueNames.get(i));
            nameToLayerIndex.put(keyPastName, i);
            nameToLayerIndex.put(valPastName, i);
            nameIsKey.put(keyPastName, true);
            nameIsKey.put(valPastName, false);
            allPastNames.add(keyPastName);
            allPastNames.add(valPastName);
        }

        String firstPresentName = kvNames.keyNames.get(0);
        INDArray firstKv = prefillOutputs.get(firstPresentName);
        if (firstKv == null) {
            firstKv = prefillOutputs.get(ioConfig.presentToInputName(firstPresentName));
        }
        if (firstKv == null) {
            throw new IllegalStateException("No prefill output found for " + firstPresentName);
        }

        this.numHeads = firstKv.size(1);
        this.headDim = firstKv.size(3);
        this.kvDataType = firstKv.dataType();

        this.layerCaches = new PagedKVCache[numLayers];
        for (int i = 0; i < numLayers; i++) {
            layerCaches[i] = new PagedKVCache(1, (int) maxKvLen, (int) numHeads, (int) headDim,
                    blockSize, kvDataType, 1.0);
        }

        for (int layer = 0; layer < numLayers; layer++) {
            INDArray prefillKey = prefillOutputs.get(kvNames.keyNames.get(layer));
            INDArray prefillVal = prefillOutputs.get(kvNames.valueNames.get(layer));
            if (prefillKey == null || prefillVal == null) continue;

            layerCaches[layer].append(0,
                    prefillKey.permute(0, 2, 1, 3),
                    prefillVal.permute(0, 2, 1, 3));
        }

        Nd4j.getExecutioner().commit();

        for (String pastName : allPastNames) {
            INDArray buf = Nd4j.zeros(kvDataType, 1, numHeads, maxKvLen, headDim);
            buf.setCloseable(false);
            staticKvBuffers.put(pastName, buf);
        }

        reconstructPagedBuffers(prefillSeqLen);
        Nd4j.getExecutioner().commit();
    }

    private void initPagedEmpty(SameDiff decoder, List<String> pastInputNames, long hiddenSize) {
        this.nameToLayerIndex = new HashMap<>();
        this.nameIsKey = new HashMap<>();
        this.allPastNames = new ArrayList<>();
        this.staticKvBuffers = new HashMap<>();

        this.numLayers = pastInputNames.size() / 2;
        if (numLayers == 0) return;

        String firstName = pastInputNames.get(0);
        INDArray empty = inferEmptyKvBuffer(decoder, firstName, 1, hiddenSize);
        this.numHeads = empty.size(1);
        this.headDim = empty.size(3);
        this.kvDataType = empty.dataType();
        empty.close();

        this.layerCaches = new PagedKVCache[numLayers];
        int layerIdx = 0;
        for (int i = 0; i < pastInputNames.size(); i += 2) {
            String keyName = pastInputNames.get(i);
            String valName = pastInputNames.get(i + 1);

            nameToLayerIndex.put(keyName, layerIdx);
            nameToLayerIndex.put(valName, layerIdx);
            nameIsKey.put(keyName, true);
            nameIsKey.put(valName, false);
            allPastNames.add(keyName);
            allPastNames.add(valName);

            layerCaches[layerIdx] = new PagedKVCache(1, (int) maxKvLen, (int) numHeads,
                    (int) headDim, blockSize, kvDataType, 1.0);

            INDArray buf1 = Nd4j.zeros(kvDataType, 1, numHeads, maxKvLen, headDim);
            buf1.setCloseable(false);
            staticKvBuffers.put(keyName, buf1);

            INDArray buf2 = Nd4j.zeros(kvDataType, 1, numHeads, maxKvLen, headDim);
            buf2.setCloseable(false);
            staticKvBuffers.put(valName, buf2);

            layerIdx++;
        }
    }

    private void scatterPagedEntries(Map<String, INDArray> decoderOutputs,
                                      ModelIOConfig.KVCacheNames kvNames,
                                      int numEntries) {
        for (int i = 0; i < numEntries; i++) {
            for (int layer = 0; layer < numLayers; layer++) {
                String keyPresentName = kvNames.keyNames.get(layer);
                String valPresentName = kvNames.valueNames.get(layer);

                INDArray presentKey = decoderOutputs.get(keyPresentName);
                INDArray presentVal = decoderOutputs.get(valPresentName);
                if (presentKey == null || presentVal == null) continue;

                long srcPos = presentKey.size(2) - numEntries + i;
                INDArray keySlice = presentKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(srcPos), NDArrayIndex.all());
                INDArray valSlice = presentVal.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(srcPos), NDArrayIndex.all());

                layerCaches[layer].append(0,
                        keySlice.reshape(1, 1, numHeads, headDim),
                        valSlice.reshape(1, 1, numHeads, headDim));

                String keyPastName = ioConfig.presentToInputName(keyPresentName);
                String valPastName = ioConfig.presentToInputName(valPresentName);

                INDArray keyBuf = staticKvBuffers.get(keyPastName);
                if (keyBuf != null) {
                    keyBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.point(cachePosition + i), NDArrayIndex.all())
                            .assign(keySlice);
                }
                INDArray valBuf = staticKvBuffers.get(valPastName);
                if (valBuf != null) {
                    valBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.point(cachePosition + i), NDArrayIndex.all())
                            .assign(valSlice);
                }
            }
        }
    }

    private void reconstructPagedBuffers(long validLen) {
        if (layerCaches == null || allPastNames == null) return;

        for (int layer = 0; layer < numLayers; layer++) {
            PagedKVCache cache = layerCaches[layer];
            int[] rawPageTable = cache.getRawPageTable(0);

            for (String pastName : allPastNames) {
                Integer idx = nameToLayerIndex.get(pastName);
                if (idx == null || idx != layer) continue;

                boolean isKey = nameIsKey.get(pastName);
                INDArray blockPool = isKey ? cache.getKeyBlockPool() : cache.getValueBlockPool();
                INDArray buf = staticKvBuffers.get(pastName);
                if (buf == null) continue;

                int pos = 0;
                while (pos < validLen) {
                    int logicalBlock = pos / blockSize;
                    int offsetInBlock = pos % blockSize;
                    int physicalBlock = rawPageTable[logicalBlock];
                    if (physicalBlock < 0) { pos++; continue; }

                    int tokensInBlock = (int) Math.min(blockSize - offsetInBlock, validLen - pos);

                    INDArray src = blockPool.get(NDArrayIndex.point(physicalBlock),
                            NDArrayIndex.interval(offsetInBlock, offsetInBlock + tokensInBlock),
                            NDArrayIndex.all(), NDArrayIndex.all());
                    INDArray dest = buf.get(NDArrayIndex.point(0), NDArrayIndex.all(),
                            NDArrayIndex.interval(pos, pos + tokensInBlock), NDArrayIndex.all());
                    dest.assign(src.permute(1, 0, 2));

                    pos += tokensInBlock;
                }
            }
        }
    }

    // ══════════════════════════════════════════════════════════════
    //  Utility methods
    // ══════════════════════════════════════════════════════════════

    private INDArray findPresentOutput(Map<String, INDArray> outputs, String presentName, String pastName) {
        INDArray result = outputs.get(presentName);
        if (result == null) {
            for (String outputKey : outputs.keySet()) {
                if (outputKey.contains(pastName.replace(ioConfig.getKvCachePrefix(), ""))) {
                    result = outputs.get(outputKey);
                    break;
                }
            }
        }
        return result;
    }

    private INDArray generateOrthogonalMatrix(int d, Random rng) {
        float[] data = new float[d * d];
        for (int i = 0; i < data.length; i++) data[i] = (float) rng.nextGaussian();
        INDArray G = Nd4j.createFromArray(data).reshape(d, d);

        float[][] Q = new float[d][d];
        float[][] cols = new float[d][d];
        for (int j = 0; j < d; j++)
            for (int i = 0; i < d; i++)
                cols[j][i] = G.getFloat(i, j);

        for (int j = 0; j < d; j++) {
            float[] v = new float[d];
            System.arraycopy(cols[j], 0, v, 0, d);
            for (int k = 0; k < j; k++) {
                float dot = 0;
                for (int i = 0; i < d; i++) dot += v[i] * Q[k][i];
                for (int i = 0; i < d; i++) v[i] -= dot * Q[k][i];
            }
            float norm = 0;
            for (int i = 0; i < d; i++) norm += v[i] * v[i];
            norm = (float) Math.sqrt(norm);
            if (norm > 1e-10) for (int i = 0; i < d; i++) v[i] /= norm;
            Q[j] = v;
        }

        float[] result = new float[d * d];
        for (int i = 0; i < d; i++) System.arraycopy(Q[i], 0, result, i * d, d);
        G.close();
        return Nd4j.createFromArray(result).reshape(d, d);
    }

    private INDArray generateGaussianMatrix(int rows, int cols, Random rng) {
        float[] data = new float[rows * cols];
        for (int i = 0; i < data.length; i++) data[i] = (float) rng.nextGaussian();
        return Nd4j.createFromArray(data).reshape(rows, cols);
    }

    // ══════════════════════════════════════════════════════════════
    //  Cleanup
    // ══════════════════════════════════════════════════════════════

    @Override
    public void close() {
        closeBufferMap(staticKvBuffers);
        closeBufferMap(quantizedBuffers);
        closeBufferMap(scaleBuffers);
        closeBufferMap(rotationMatrices);
        closeBufferMap(qjlMatrices);
        closeBufferMap(keyMseReconstruction);
        closeBufferMap(keyQjlSigns);
        closeBufferMap(keyResidualNorms);
        closeBufferMap(valueMseIndices);
        closeBufferMap(valueVecNorms);

        if (layerCaches != null) {
            for (PagedKVCache cache : layerCaches) {
                if (cache != null) cache.close();
            }
            layerCaches = null;
        }

        staticKvBuffers = null;
        quantizedBuffers = null;
        scaleBuffers = null;
        initialized = false;
        log.info("UnifiedKvCacheManager[{}] closed", strategy);
    }

    private void closeBufferMap(Map<String, INDArray> buffers) {
        if (buffers != null) {
            for (INDArray buf : buffers.values()) {
                if (buf != null && !buf.wasClosed()) {
                    SameDiffMemoryUtils.safeClose(buf);
                }
            }
        }
    }

    @Override
    public String toString() {
        return String.format("UnifiedKvCacheManager[strategy=%s, maxKvLen=%d, cachePos=%d, "
                        + "inGraphScatter=%s, initialized=%s]",
                strategy, maxKvLen, cachePosition, inGraphScatterEnabled, initialized);
    }
}
