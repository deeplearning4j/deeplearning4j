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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Paged KV cache manager using block-based allocation via {@link PagedKVCache}.
 *
 * <p>Instead of pre-allocating dense [batch, heads, maxKvLen, headDim] buffers for every
 * layer upfront (which can OOM on memory-constrained GPUs), this manager allocates
 * fixed-size blocks on demand as tokens are generated. Memory grows incrementally
 * rather than being fully committed at prefill time.</p>
 *
 * <h3>Architecture</h3>
 * <p>One {@link PagedKVCache} per decoder layer. Each cache has its own key and value
 * block pools with shared page tables (since all layers process the same sequences).
 * For prepareInputs, contiguous KV tensors are reconstructed from blocks for the
 * decoder's standard attention computation.</p>
 *
 * <h3>Memory advantage</h3>
 * <p>For a model with 24 layers and prefill=2073 + maxNewTokens=4096:</p>
 * <ul>
 *   <li>Static: Pre-allocates all 6169 positions across all layers upfront (~4GB in FP32)</li>
 *   <li>Paged: Only allocates blocks as tokens arrive. At decode step N, only
 *       ceil(prefill+N / blockSize) blocks are allocated per layer.</li>
 * </ul>
 *
 * @see KvCacheManager
 * @see PagedKVCache
 */
@Slf4j
public class PagedKvCacheManager implements KvCacheManager {

    private final int blockSize;

    /** Per-layer PagedKVCache instances. Key: layer index (0-based). */
    private PagedKVCache[] layerCaches;

    /** Mapping from past_key_values input name to layer index. */
    private Map<String, Integer> nameToLayerIndex;

    /** Whether each name is a key (true) or value (false). */
    private Map<String, Boolean> nameIsKey;

    /** Ordered list of all past KV input names (keys then values). */
    private List<String> allPastNames;

    /** Pre-allocated contiguous buffers for prepareInputs (reused across steps). */
    private Map<String, INDArray> contiguousBuffers;

    @Getter
    private long maxKvLen;

    @Getter
    private long cachePosition;

    private boolean initialized;

    private int numLayers;
    private long numKvHeads;
    private long headDim;
    private DataType kvDataType;

    private final ModelIOConfig ioConfig;

    public PagedKvCacheManager() {
        this(PagedKVCache.DEFAULT_BLOCK_SIZE);
    }

    public PagedKvCacheManager(int blockSize) {
        this(blockSize, ModelIOConfig.builder().build());
    }

    public PagedKvCacheManager(int blockSize, ModelIOConfig ioConfig) {
        this.blockSize = blockSize;
        this.ioConfig = ioConfig;
        this.initialized = false;
        this.cachePosition = 0;
        this.maxKvLen = -1;
    }

    @Override
    public KvCacheStrategy getStrategy() {
        return KvCacheStrategy.PAGED;
    }

    @Override
    public void initializeFromPrefill(Map<String, INDArray> prefillOutputs,
                                       DecoderUtils.KVCacheNames kvNames,
                                       int maxNewTokens, long prefillSeqLen) {
        this.maxKvLen = prefillSeqLen + maxNewTokens;
        this.numLayers = kvNames.keyNames.size();

        log.info("  PagedKvCacheManager: prefillLen={}, maxKvLen={}, blockSize={}, "
                        + "{} layers (key + value)",
                prefillSeqLen, maxKvLen, blockSize, numLayers);

        // Build name mappings
        this.nameToLayerIndex = new HashMap<>();
        this.nameIsKey = new HashMap<>();
        this.allPastNames = new ArrayList<>();

        for (int i = 0; i < numLayers; i++) {
            String keyPresentName = kvNames.keyNames.get(i);
            String valPresentName = kvNames.valueNames.get(i);
            String keyPastName = ioConfig.presentToInputName(keyPresentName);
            String valPastName = ioConfig.presentToInputName(valPresentName);

            nameToLayerIndex.put(keyPastName, i);
            nameToLayerIndex.put(valPastName, i);
            nameIsKey.put(keyPastName, true);
            nameIsKey.put(valPastName, false);
            allPastNames.add(keyPastName);
            allPastNames.add(valPastName);
        }

        // Infer dimensions from first prefill output
        String firstPresentName = kvNames.keyNames.get(0);
        INDArray firstKv = prefillOutputs.get(firstPresentName);
        if (firstKv == null) {
            // Try input name
            firstKv = prefillOutputs.get(ioConfig.presentToInputName(firstPresentName));
        }
        if (firstKv == null) {
            throw new IllegalStateException("No prefill output found for " + firstPresentName);
        }

        // Prefill KV shape: [batch=1, heads, seqLen, headDim]
        this.numKvHeads = firstKv.size(1);
        this.headDim = firstKv.size(3);
        this.kvDataType = firstKv.dataType();

        // Create per-layer PagedKVCache instances
        this.layerCaches = new PagedKVCache[numLayers];
        for (int i = 0; i < numLayers; i++) {
            layerCaches[i] = new PagedKVCache(
                    1, // batch size
                    (int) maxKvLen,
                    (int) numKvHeads,
                    (int) headDim,
                    blockSize,
                    kvDataType,
                    1.0 // no headroom needed for single sequence
            );
        }

        // Append prefill KV data into paged caches
        for (int layer = 0; layer < numLayers; layer++) {
            String keyPresentName = kvNames.keyNames.get(layer);
            String valPresentName = kvNames.valueNames.get(layer);
            INDArray prefillKey = prefillOutputs.get(keyPresentName);
            INDArray prefillVal = prefillOutputs.get(valPresentName);

            if (prefillKey == null || prefillVal == null) {
                log.warn("Missing prefill KV for layer {}", layer);
                continue;
            }

            // Prefill shape: [1, heads, seqLen, headDim] -> need [1, seqLen, heads, headDim]
            // PagedKVCache.append expects [1, newLen, numKvHeads, headDim]
            INDArray keyTransposed = prefillKey.permute(0, 2, 1, 3);
            INDArray valTransposed = prefillVal.permute(0, 2, 1, 3);

            layerCaches[layer].append(0, keyTransposed, valTransposed);
        }

        // Commit after all layer appends to flush CUDA operations before allocating contiguous buffers
        Nd4j.getExecutioner().commit();

        // Allocate contiguous buffers for prepareInputs
        this.contiguousBuffers = new HashMap<>();
        for (String pastName : allPastNames) {
            INDArray buf = Nd4j.zeros(kvDataType, 1, numKvHeads, maxKvLen, headDim);
            buf.setCloseable(false);
            contiguousBuffers.put(pastName, buf);
        }

        // Fill contiguous buffers with prefill data
        reconstructContiguousBuffers(prefillSeqLen);

        Nd4j.getExecutioner().commit();
        this.cachePosition = prefillSeqLen;
        this.initialized = true;

        long totalBlockPoolMB = 0;
        for (PagedKVCache cache : layerCaches) {
            // key + value pools
            totalBlockPoolMB += 2L * cache.getNumBlocks() * blockSize * numKvHeads * headDim * kvDataType.width();
        }
        log.info("  PagedKvCacheManager: initialized with {} layers, block pools ~{} MB, "
                        + "contiguous buffers ~{} MB",
                numLayers, totalBlockPoolMB / (1024 * 1024),
                contiguousBuffers.size() * numKvHeads * maxKvLen * headDim * kvDataType.width() / (1024 * 1024));
    }

    @Override
    public void initializeEmpty(SameDiff decoder, List<String> decoderInputNames,
                                 int maxKvLen, long hiddenSize) {
        this.maxKvLen = maxKvLen;
        this.nameToLayerIndex = new HashMap<>();
        this.nameIsKey = new HashMap<>();
        this.allPastNames = new ArrayList<>();
        this.contiguousBuffers = new HashMap<>();

        List<String> pastInputNames = new ArrayList<>();
        for (String inputName : decoderInputNames) {
            if (ioConfig.isKvCacheInput(inputName)) {
                pastInputNames.add(inputName);
            }
        }

        // Infer layer count and dimensions
        // Past names come in pairs: key_0, value_0, key_1, value_1, ...
        this.numLayers = pastInputNames.size() / 2;
        if (numLayers == 0) {
            log.warn("No KV cache inputs found in decoder");
            this.initialized = true;
            this.cachePosition = 0;
            return;
        }

        // Infer dimensions from first input
        String firstName = pastInputNames.get(0);
        INDArray empty = DecoderUtils.createEmptyKvCache(decoder, firstName, 1, hiddenSize);
        this.numKvHeads = empty.size(1);
        this.headDim = empty.size(3);
        this.kvDataType = empty.dataType();
        empty.close();

        // Create layer caches
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

            layerCaches[layerIdx] = new PagedKVCache(
                    1, (int) maxKvLen, (int) numKvHeads, (int) headDim,
                    blockSize, kvDataType, 1.0);

            INDArray buf1 = Nd4j.zeros(kvDataType, 1, numKvHeads, maxKvLen, headDim);
            buf1.setCloseable(false);
            contiguousBuffers.put(keyName, buf1);

            INDArray buf2 = Nd4j.zeros(kvDataType, 1, numKvHeads, maxKvLen, headDim);
            buf2.setCloseable(false);
            contiguousBuffers.put(valName, buf2);

            layerIdx++;
        }

        this.cachePosition = 0;
        this.initialized = true;
        log.info("  PagedKvCacheManager: initialized empty with maxKvLen={}, {} layers", maxKvLen, numLayers);
    }

    @Override
    public void prepareInputs(Map<String, INDArray> decoderInputMap, SameDiff decoder,
                               long hiddenSize, boolean dspActive) {
        if (!initialized || contiguousBuffers == null) return;

        for (String inputName : decoder.inputs()) {
            if (!ioConfig.isKvCacheInput(inputName)) continue;

            INDArray buf = contiguousBuffers.get(inputName);
            if (buf != null) {
                if (dspActive) {
                    // Padded mode: pass full buffer (fixed shape for potential CUDA graphs)
                    decoderInputMap.put(inputName, buf);
                } else {
                    // View mode: pass [0:cachePos] view
                    if (cachePosition > 0 && cachePosition < buf.size(2)) {
                        INDArray view = buf.get(
                                NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(0, cachePosition), NDArrayIndex.all());
                        decoderInputMap.put(inputName, view);
                    } else if (cachePosition >= buf.size(2)) {
                        decoderInputMap.put(inputName, buf);
                    } else {
                        decoderInputMap.put(inputName,
                                DecoderUtils.createEmptyKvCache(decoder, inputName, 1, hiddenSize));
                    }
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
        for (int layer = 0; layer < numLayers; layer++) {
            String keyPresentName = kvNames.keyNames.get(layer);
            String valPresentName = kvNames.valueNames.get(layer);

            INDArray presentKey = decoderOutputs.get(keyPresentName);
            INDArray presentVal = decoderOutputs.get(valPresentName);

            if (presentKey == null || presentVal == null) continue;

            // Extract the new token's KV entry (last position in seq dim)
            // Present shape: [1, heads, seqLen, headDim] where new entry is at last pos
            long seqLen = presentKey.size(2);
            INDArray newKeySlice = presentKey.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(seqLen - 1), NDArrayIndex.all());
            INDArray newValSlice = presentVal.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(seqLen - 1), NDArrayIndex.all());

            // Reshape for PagedKVCache: [1, 1, heads, headDim]
            INDArray newKey4d = newKeySlice.reshape(1, 1, numKvHeads, headDim);
            INDArray newVal4d = newValSlice.reshape(1, 1, numKvHeads, headDim);

            layerCaches[layer].append(0, newKey4d, newVal4d);

            // Update contiguous buffer at cachePosition
            String keyPastName = ioConfig.presentToInputName(keyPresentName);
            String valPastName = ioConfig.presentToInputName(valPresentName);

            INDArray keyBuf = contiguousBuffers.get(keyPastName);
            INDArray valBuf = contiguousBuffers.get(valPastName);

            if (keyBuf != null) {
                // Write [1, heads, 1, headDim] into contiguous buffer at cachePosition
                INDArray keyDest = keyBuf.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(cachePosition), NDArrayIndex.all());
                keyDest.assign(newKeySlice);
            }
            if (valBuf != null) {
                INDArray valDest = valBuf.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(cachePosition), NDArrayIndex.all());
                valDest.assign(newValSlice);
            }
        }

        cachePosition++;
    }

    @Override
    public void scatterMultipleEntries(Map<String, INDArray> decoderOutputs,
                                        DecoderUtils.KVCacheNames kvNames,
                                        int numEntries) {
        if (numEntries <= 0) return;

        for (int i = 0; i < numEntries; i++) {
            // For speculative decode, we scatter one at a time
            // The decoder output contains multiple positions
            for (int layer = 0; layer < numLayers; layer++) {
                String keyPresentName = kvNames.keyNames.get(layer);
                String valPresentName = kvNames.valueNames.get(layer);

                INDArray presentKey = decoderOutputs.get(keyPresentName);
                INDArray presentVal = decoderOutputs.get(valPresentName);
                if (presentKey == null || presentVal == null) continue;

                // Extract position i from the multi-token output
                long srcPos = presentKey.size(2) - numEntries + i;
                INDArray keySlice = presentKey.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(srcPos), NDArrayIndex.all());
                INDArray valSlice = presentVal.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(srcPos), NDArrayIndex.all());

                INDArray newKey4d = keySlice.reshape(1, 1, numKvHeads, headDim);
                INDArray newVal4d = valSlice.reshape(1, 1, numKvHeads, headDim);
                layerCaches[layer].append(0, newKey4d, newVal4d);

                String keyPastName = ioConfig.presentToInputName(keyPresentName);
                String valPastName = ioConfig.presentToInputName(valPresentName);

                INDArray keyBuf = contiguousBuffers.get(keyPastName);
                INDArray valBuf = contiguousBuffers.get(valPastName);

                if (keyBuf != null) {
                    INDArray keyDest = keyBuf.get(
                            NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.point(cachePosition + i), NDArrayIndex.all());
                    keyDest.assign(keySlice);
                }
                if (valBuf != null) {
                    INDArray valDest = valBuf.get(
                            NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.point(cachePosition + i), NDArrayIndex.all());
                    valDest.assign(valSlice);
                }
            }
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
        // Paged cache reconstructs contiguous buffers each step — addresses are stable
        // so CUDA graph replay is possible in padded mode
        return true;
    }

    @Override
    public Map<String, INDArray> getStaticKvBuffers() {
        return contiguousBuffers;
    }

    @Override
    public void close() {
        if (layerCaches != null) {
            for (PagedKVCache cache : layerCaches) {
                if (cache != null) {
                    cache.close();
                }
            }
            layerCaches = null;
        }
        if (contiguousBuffers != null) {
            for (INDArray buf : contiguousBuffers.values()) {
                if (buf != null && !buf.wasClosed()) {
                    buf.setCloseable(true);
                    buf.close();
                }
            }
            contiguousBuffers = null;
        }
        initialized = false;
        log.info("PagedKvCacheManager closed");
    }

    /**
     * Reconstruct contiguous KV buffers from paged block storage.
     * Called during initialization to populate buffers with prefill data.
     *
     * <p>Uses block-level bulk copies instead of per-token copies to minimize
     * CUDA kernel launches. For prefillLen=1679 with blockSize=64, this reduces
     * kernel launches from ~1679 per buffer to ~27 per buffer.</p>
     */
    private void reconstructContiguousBuffers(long validLen) {
        int blocksCopied = 0;
        for (int layer = 0; layer < numLayers; layer++) {
            PagedKVCache cache = layerCaches[layer];
            int[] rawPageTable = cache.getRawPageTable(0);

            for (String pastName : allPastNames) {
                Integer idx = nameToLayerIndex.get(pastName);
                if (idx == null || idx != layer) continue;

                boolean isKey = nameIsKey.get(pastName);
                INDArray blockPool = isKey ? cache.getKeyBlockPool() : cache.getValueBlockPool();
                INDArray buf = contiguousBuffers.get(pastName);
                if (buf == null) continue;

                // Copy from paged blocks into contiguous buffer using block-level bulk ops.
                // Block pool: [numBlocks, blockSize, numKvHeads, headDim]
                // Contiguous: [1, numKvHeads, validLen, headDim]
                int pos = 0;
                while (pos < validLen) {
                    int logicalBlock = pos / blockSize;
                    int offsetInBlock = pos % blockSize;
                    int physicalBlock = rawPageTable[logicalBlock];
                    if (physicalBlock < 0) {
                        pos++;
                        continue;
                    }

                    // How many tokens in this block to copy?
                    int tokensInBlock = (int) Math.min(blockSize - offsetInBlock, validLen - pos);

                    // Source: blockPool[physicalBlock, offsetInBlock..offsetInBlock+tokensInBlock, :, :]
                    // Shape: [tokensInBlock, numKvHeads, headDim]
                    INDArray src = blockPool.get(
                            NDArrayIndex.point(physicalBlock),
                            NDArrayIndex.interval(offsetInBlock, offsetInBlock + tokensInBlock),
                            NDArrayIndex.all(), NDArrayIndex.all());

                    // Dest: buf[0, :, pos..pos+tokensInBlock, :]
                    // Shape: [numKvHeads, tokensInBlock, headDim]
                    INDArray dest = buf.get(
                            NDArrayIndex.point(0), NDArrayIndex.all(),
                            NDArrayIndex.interval(pos, pos + tokensInBlock), NDArrayIndex.all());

                    // src is [tokensInBlock, heads, headDim], dest is [heads, tokensInBlock, headDim]
                    // Need to permute src to [heads, tokensInBlock, headDim] before assign
                    dest.assign(src.permute(1, 0, 2));

                    pos += tokensInBlock;
                    blocksCopied++;

                    // Periodic CUDA commit to prevent driver memory accumulation
                    if (blocksCopied % 8 == 0) {
                        Nd4j.getExecutioner().commit();
                    }
                }
            }
        }

        // Final commit to flush all pending operations
        if (blocksCopied > 0) {
            Nd4j.getExecutioner().commit();
        }
    }
}
