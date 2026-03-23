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

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheDequantize;
import org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheQuantize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Quantized KV cache manager that stores KV data in compressed format (INT8/FP8/INT4).
 *
 * <p>Uses static pre-allocated buffers (like {@link StaticKvCacheManager}) but stores
 * KV entries in quantized format with per-head-row scale factors. This reduces
 * memory usage by 2-4x compared to full-precision storage, enabling longer context
 * lengths within the same GPU memory budget.</p>
 *
 * <h3>Architecture</h3>
 * <p>Unlike {@link QuantizedPagedKVCache} which uses paged block allocation, this
 * manager uses dense static buffers compatible with the existing decode loop. The
 * quantization/dequantization is handled transparently:</p>
 * <ul>
 *   <li><b>On scatter</b>: New KV entries from the decoder are quantized via the native
 *       {@code kv_cache_quantize} op before being written to the quantized buffers.</li>
 *   <li><b>On prepareInputs</b>: The quantized buffers are dequantized via the native
 *       {@code kv_cache_dequantize} op to produce full-precision KV tensors for the
 *       decoder's attention computation.</li>
 * </ul>
 *
 * <h3>Quantization scheme</h3>
 * <p>Per-row symmetric quantization: each row (one token position across headDim values)
 * gets its own scale factor = max(abs(row)) / 127 (for INT8). This preserves accuracy
 * for typical attention value distributions while achieving 2-4x compression.</p>
 *
 * <h3>Limitations</h3>
 * <ul>
 *   <li>CUDA graph replay is NOT supported because dequantized buffers are regenerated
 *       each step (changing tensor addresses). Future work: pre-allocate fixed dequant
 *       buffers to enable CUDA graph compatibility.</li>
 *   <li>There is a per-step compute cost for dequantization. For memory-bound models
 *       (large KV caches), this is a net win because reduced memory traffic outweighs
 *       the dequantization ALU cost.</li>
 * </ul>
 *
 * @see KvCacheManager
 * @see QuantizedPagedKVCache
 * @see KVCacheQuantize
 * @see KVCacheDequantize
 */
@Slf4j
public class QuantizedKvCacheManager implements KvCacheManager {

    @Getter
    private final QuantizedPagedKVCache.QuantFormat quantFormat;

    /** Original data type of the model's KV values (for dequantization output). */
    private final DataType originalDataType;

    /**
     * Quantized KV buffers: per-layer quantized storage.
     * Key: past_key_values input name -> quantized INT8 buffer [1, heads, maxKvLen, headDim]
     */
    private Map<String, INDArray> quantizedBuffers;

    /**
     * Per-row scale factors: per-layer scale storage.
     * Key: past_key_values input name -> FLOAT32 scales [1, heads, maxKvLen]
     * One scale per (head, position) row.
     */
    private Map<String, INDArray> scaleBuffers;

    /**
     * Pre-allocated dequantized output buffers: reused each step to avoid allocation.
     * Key: past_key_values input name -> full-precision buffer [1, heads, maxKvLen, headDim]
     * Using fixed-address buffers means CUDA graph replay could be enabled in the future.
     */
    private Map<String, INDArray> dequantizedBuffers;

    @Getter
    private long maxKvLen;

    @Getter
    private long cachePosition;

    private boolean initialized;

    /** Number of KV heads per layer (inferred from prefill output shapes). */
    private long numHeads;

    /** Dimension per head (inferred from prefill output shapes). */
    private long headDim;

    /** Model I/O configuration for name resolution. */
    private final ModelIOConfig ioConfig;

    /**
     * Create a quantized KV cache manager.
     *
     * @param quantFormat     quantization format (INT8, FP8_E4M3, FP8_E5M2, INT4)
     * @param originalDataType the model's original KV data type (e.g., FLOAT, HALF)
     */
    public QuantizedKvCacheManager(QuantizedPagedKVCache.QuantFormat quantFormat,
                                    DataType originalDataType) {
        this(quantFormat, originalDataType, ModelIOConfig.builder().build());
    }

    /**
     * Create a quantized KV cache manager with explicit I/O configuration.
     */
    public QuantizedKvCacheManager(QuantizedPagedKVCache.QuantFormat quantFormat,
                                    DataType originalDataType,
                                    ModelIOConfig ioConfig) {
        this.quantFormat = quantFormat;
        this.originalDataType = originalDataType;
        this.ioConfig = ioConfig;
        this.initialized = false;
        this.cachePosition = 0;
        this.maxKvLen = -1;
    }

    /**
     * Create with INT8 quantization and FLOAT original type (common defaults).
     */
    public QuantizedKvCacheManager() {
        this(QuantizedPagedKVCache.QuantFormat.INT8, DataType.FLOAT);
    }

    @Override
    public KvCacheStrategy getStrategy() {
        return KvCacheStrategy.QUANTIZED;
    }

    @Override
    public void initializeFromPrefill(Map<String, INDArray> prefillOutputs,
                                      DecoderUtils.KVCacheNames kvNames,
                                      int maxNewTokens, long prefillSeqLen) {
        this.maxKvLen = prefillSeqLen + maxNewTokens;
        log.info("  QuantizedKvCacheManager: prefillLen={}, maxKvLen={}, format={}, "
                        + "{} key + {} value tensors",
                prefillSeqLen, maxKvLen, quantFormat,
                kvNames.keyNames.size(), kvNames.valueNames.size());

        this.quantizedBuffers = new HashMap<>();
        this.scaleBuffers = new HashMap<>();
        this.dequantizedBuffers = new HashMap<>();

        // Combine key and value names for uniform processing
        List<String> allNames = new ArrayList<>(kvNames.keyNames);
        allNames.addAll(kvNames.valueNames);

        for (String name : allNames) {
            // Find the corresponding present_ output name
            String presentName = ioConfig.inputToPresentName(name);
            INDArray presentKv = prefillOutputs.get(presentName);
            if (presentKv == null) {
                // Try alternate naming conventions
                for (String outputKey : prefillOutputs.keySet()) {
                    if (outputKey.contains(name.replace(ioConfig.getKvCachePrefix(), ""))) {
                        presentKv = prefillOutputs.get(outputKey);
                        break;
                    }
                }
            }

            if (presentKv == null) {
                log.warn("No prefill output found for KV name: {}", name);
                continue;
            }

            // Shape: [batch=1, heads, seqLen, headDim]
            this.numHeads = presentKv.size(1);
            this.headDim = presentKv.size(3);
            DataType kvType = presentKv.dataType();

            // Allocate quantized buffer: [1, heads, maxKvLen, headDim] in INT8
            INDArray quantBuf = Nd4j.zeros(DataType.INT8, 1, numHeads, maxKvLen, headDim);
            quantizedBuffers.put(name, quantBuf);

            // Allocate scale buffer: [1, heads, maxKvLen] in FLOAT32
            // One scale per row (a row = headDim values for one head at one position)
            INDArray scaleBuf = Nd4j.ones(DataType.FLOAT, 1, numHeads, maxKvLen);
            scaleBuffers.put(name, scaleBuf);

            // Allocate dequantized buffer: [1, heads, maxKvLen, headDim] in original dtype
            INDArray deqBuf = Nd4j.zeros(kvType, 1, numHeads, maxKvLen, headDim);
            dequantizedBuffers.put(name, deqBuf);

            // Quantize and store the prefill data
            quantizeAndStore(presentKv, name, 0, prefillSeqLen);
        }

        Nd4j.getExecutioner().commit();
        this.cachePosition = prefillSeqLen;
        this.initialized = true;

        long quantizedMB = 0;
        long fullPrecMB = 0;
        for (INDArray buf : quantizedBuffers.values()) {
            quantizedMB += buf.length(); // INT8 = 1 byte each
        }
        for (INDArray buf : scaleBuffers.values()) {
            quantizedMB += buf.length() * 4; // FLOAT32 = 4 bytes each
        }
        for (INDArray buf : dequantizedBuffers.values()) {
            fullPrecMB += buf.length() * buf.dataType().width();
        }
        log.info("  QuantizedKvCacheManager: quantized storage ~{} MB, "
                        + "dequant buffers ~{} MB, vs ~{} MB full-precision static",
                quantizedMB / (1024 * 1024),
                fullPrecMB / (1024 * 1024),
                fullPrecMB / (1024 * 1024));
    }

    @Override
    public void initializeEmpty(SameDiff decoder, List<String> decoderInputNames,
                                int maxKvLen, long hiddenSize) {
        this.maxKvLen = maxKvLen;
        this.quantizedBuffers = new HashMap<>();
        this.scaleBuffers = new HashMap<>();
        this.dequantizedBuffers = new HashMap<>();

        List<String> pastInputNames = new ArrayList<>();
        for (String inputName : decoderInputNames) {
            if (ioConfig.isKvCacheInput(inputName)) {
                pastInputNames.add(inputName);
            }
        }

        for (String pastInputName : pastInputNames) {
            INDArray empty = DecoderUtils.createEmptyKvCache(decoder, pastInputName, 1, hiddenSize);
            long numH = empty.size(1);
            long hDim = empty.size(3);
            DataType kvType = empty.dataType();
            empty.close();

            this.numHeads = numH;
            this.headDim = hDim;

            quantizedBuffers.put(pastInputName,
                    Nd4j.zeros(DataType.INT8, 1, numH, maxKvLen, hDim));
            scaleBuffers.put(pastInputName,
                    Nd4j.ones(DataType.FLOAT, 1, numH, maxKvLen));
            dequantizedBuffers.put(pastInputName,
                    Nd4j.zeros(kvType, 1, numH, maxKvLen, hDim));
        }

        this.cachePosition = 0;
        this.initialized = true;
        log.info("  QuantizedKvCacheManager: initialized empty with maxKvLen={}, "
                        + "format={}, {} buffers",
                maxKvLen, quantFormat, quantizedBuffers.size());
    }

    @Override
    public void prepareInputs(Map<String, INDArray> decoderInputMap, SameDiff decoder,
                              long hiddenSize, boolean dspActive) {
        if (!initialized || quantizedBuffers == null) return;

        for (String inputName : decoder.inputs()) {
            if (!ioConfig.isKvCacheInput(inputName)) continue;

            INDArray quantBuf = quantizedBuffers.get(inputName);
            INDArray scaleBuf = scaleBuffers.get(inputName);
            INDArray deqBuf = dequantizedBuffers.get(inputName);

            if (quantBuf != null && scaleBuf != null && deqBuf != null) {
                if (cachePosition > 0) {
                    // Dequantize the valid region [0:cachePosition] into the dequantized buffer
                    dequantizeRegion(inputName, 0, cachePosition);

                    if (dspActive) {
                        // Padded mode: pass full dequantized buffer (zero-padded beyond cachePosition)
                        decoderInputMap.put(inputName, deqBuf);
                    } else {
                        // View mode: pass only the valid region
                        INDArray view = deqBuf.get(
                                NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(0, cachePosition), NDArrayIndex.all());
                        decoderInputMap.put(inputName, view);
                    }
                } else {
                    decoderInputMap.put(inputName,
                            DecoderUtils.createEmptyKvCache(decoder, inputName, 1, hiddenSize));
                }
            } else {
                decoderInputMap.put(inputName,
                        DecoderUtils.createEmptyKvCache(decoder, inputName, 1, hiddenSize));
            }
        }
    }

    @Override
    public void scatterNewEntries(Map<String, INDArray> decoderOutputs,
                                  DecoderUtils.KVCacheNames kvNames) {
        // Quantize and scatter each present KV output into the quantized buffers
        scatterEntries(decoderOutputs, kvNames, 1);
    }

    @Override
    public void scatterMultipleEntries(Map<String, INDArray> decoderOutputs,
                                       DecoderUtils.KVCacheNames kvNames,
                                       int numEntries) {
        if (numEntries <= 0) return;
        scatterEntries(decoderOutputs, kvNames, numEntries);
    }

    /**
     * Internal: scatter numEntries new KV entries from decoder outputs into quantized buffers.
     */
    private void scatterEntries(Map<String, INDArray> decoderOutputs,
                                DecoderUtils.KVCacheNames kvNames,
                                int numEntries) {
        List<String> allPastNames = new ArrayList<>();
        List<String> allPresentNames = new ArrayList<>();

        for (int i = 0; i < kvNames.keyNames.size(); i++) {
            allPastNames.add(kvNames.keyNames.get(i));
            // Derive present name from past name
            String presentName = ioConfig.inputToPresentName(kvNames.keyNames.get(i));
            allPresentNames.add(presentName);
        }
        for (int i = 0; i < kvNames.valueNames.size(); i++) {
            allPastNames.add(kvNames.valueNames.get(i));
            String presentName = ioConfig.inputToPresentName(kvNames.valueNames.get(i));
            allPresentNames.add(presentName);
        }

        for (int i = 0; i < allPastNames.size(); i++) {
            String pastName = allPastNames.get(i);
            String presentName = allPresentNames.get(i);

            INDArray presentKv = decoderOutputs.get(presentName);
            if (presentKv == null) {
                // Try finding by iteration
                for (Map.Entry<String, INDArray> entry : decoderOutputs.entrySet()) {
                    if (entry.getKey().contains(pastName.replace(ioConfig.getKvCachePrefix(), ""))) {
                        presentKv = entry.getValue();
                        break;
                    }
                }
            }

            if (presentKv == null) continue;

            // Present KV shape: [1, heads, totalSeqLen, headDim]
            // Extract new entries from the end: positions [totalSeqLen - numEntries : totalSeqLen]
            long totalSeqLen = presentKv.size(2);
            long sourceStart = totalSeqLen - numEntries;
            if (sourceStart < 0) sourceStart = 0;

            // For DSP with static KV + C++ scatter, the present output has maxKvLen+1 seqLen.
            // The new entry is at the known offset (maxKvLen for single-token).
            if (totalSeqLen > maxKvLen) {
                sourceStart = maxKvLen;
            }

            INDArray newEntries = presentKv.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(sourceStart, sourceStart + numEntries),
                    NDArrayIndex.all());

            quantizeAndStore(newEntries, pastName, cachePosition, numEntries);
        }

        cachePosition += numEntries;
    }

    /**
     * Quantize KV data and store into the quantized buffer at the given position.
     *
     * @param kvData      full-precision KV data [1, heads, numTokens, headDim]
     * @param bufferName  past_key_values name (buffer map key)
     * @param startPos    position in cache to start writing
     * @param numTokens   number of tokens to write
     */
    private void quantizeAndStore(INDArray kvData, String bufferName,
                                   long startPos, long numTokens) {
        INDArray quantBuf = quantizedBuffers.get(bufferName);
        INDArray scaleBuf = scaleBuffers.get(bufferName);
        if (quantBuf == null || scaleBuf == null) return;

        // Reshape KV data for per-row quantization: [numRows, headDim]
        // where numRows = heads * numTokens
        long heads = kvData.size(1);
        long hDim = kvData.size(3);

        // Transpose to [1, numTokens, heads, headDim] then reshape to [numTokens * heads, headDim]
        // so each row is one head's data at one position
        INDArray kvReshaped = kvData.permute(0, 2, 1, 3)
                .reshape(numTokens * heads, hDim);

        // Run native quantize op
        KVCacheQuantize quantOp = new KVCacheQuantize(kvReshaped, quantFormat.getNativeCode());
        INDArray[] quantResult = Nd4j.exec(quantOp);
        INDArray quantized = quantResult[0];  // [numTokens * heads, headDim] INT8
        INDArray scales = quantResult[1];      // [numTokens * heads] FLOAT32

        // Reshape back to [1, numTokens, heads, headDim] and [1, numTokens, heads]
        INDArray quantReshaped = quantized.reshape(1, numTokens, heads, hDim).permute(0, 2, 1, 3);
        INDArray scalesReshaped = scales.reshape(1, numTokens, heads).permute(0, 2, 1);

        // Write into destination buffers at [0, :, startPos:startPos+numTokens, :]
        INDArray quantDest = quantBuf.get(
                NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(startPos, startPos + numTokens), NDArrayIndex.all());
        quantDest.assign(quantReshaped);

        INDArray scaleDest = scaleBuf.get(
                NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(startPos, startPos + numTokens));
        scaleDest.assign(scalesReshaped);
    }

    /**
     * Dequantize a region of the quantized buffer into the dequantized buffer.
     *
     * @param bufferName past_key_values name
     * @param startPos   start position (inclusive)
     * @param endPos     end position (exclusive)
     */
    private void dequantizeRegion(String bufferName, long startPos, long endPos) {
        INDArray quantBuf = quantizedBuffers.get(bufferName);
        INDArray scaleBuf = scaleBuffers.get(bufferName);
        INDArray deqBuf = dequantizedBuffers.get(bufferName);
        if (quantBuf == null || scaleBuf == null || deqBuf == null) return;

        long numTokens = endPos - startPos;
        if (numTokens <= 0) return;

        // Extract the region to dequantize
        INDArray quantRegion = quantBuf.get(
                NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(startPos, endPos), NDArrayIndex.all());
        INDArray scaleRegion = scaleBuf.get(
                NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(startPos, endPos));

        // Reshape for per-row dequantization: [numRows, headDim]
        long heads = quantRegion.size(1);
        long hDim = quantRegion.size(3);

        INDArray quantFlat = quantRegion.permute(0, 2, 1, 3)
                .reshape(numTokens * heads, hDim);
        INDArray scaleFlat = scaleRegion.permute(0, 2, 1)
                .reshape(numTokens * heads);

        // Run native dequantize op
        KVCacheDequantize deqOp = new KVCacheDequantize(quantFlat, scaleFlat, quantFormat.getNativeCode());
        INDArray[] deqResult = Nd4j.exec(deqOp);
        INDArray dequantized = deqResult[0]; // [numTokens * heads, headDim] FLOAT32

        // Reshape back and write into dequantized buffer
        INDArray deqReshaped = dequantized.reshape(1, numTokens, heads, hDim).permute(0, 2, 1, 3);

        // Cast to original dtype if needed
        if (deqBuf.dataType() != dequantized.dataType()) {
            INDArray casted = deqReshaped.castTo(deqBuf.dataType());
            INDArray deqDest = deqBuf.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(startPos, endPos), NDArrayIndex.all());
            deqDest.assign(casted);
        } else {
            INDArray deqDest = deqBuf.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(startPos, endPos), NDArrayIndex.all());
            deqDest.assign(deqReshaped);
        }
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
        // Quantized KV cache uses pre-allocated dequant buffers at fixed addresses,
        // so CUDA graph replay could work in padded mode. However, the dequantize
        // op itself is not yet captured in the graph, so return false for now.
        return false;
    }

    @Override
    public Map<String, INDArray> getStaticKvBuffers() {
        // Return the dequantized buffers as the "static" buffers for DSP compatibility.
        // DSP's configureKvCacheRetention needs full-precision buffers.
        return dequantizedBuffers;
    }

    @Override
    public void close() {
        closeBufferMap(quantizedBuffers);
        closeBufferMap(scaleBuffers);
        closeBufferMap(dequantizedBuffers);
        quantizedBuffers = null;
        scaleBuffers = null;
        dequantizedBuffers = null;
        initialized = false;
        log.info("QuantizedKvCacheManager closed");
    }

    private void closeBufferMap(Map<String, INDArray> buffers) {
        if (buffers != null) {
            for (INDArray buf : buffers.values()) {
                if (buf != null && !buf.wasClosed()) {
                    buf.setCloseable(true);
                    buf.close();
                }
            }
        }
    }

    @Override
    public String toString() {
        return String.format("QuantizedKvCacheManager[format=%s, maxKvLen=%d, cachePos=%d, "
                        + "heads=%d, headDim=%d, initialized=%s]",
                quantFormat, maxKvLen, cachePosition, numHeads, headDim, initialized);
    }
}
