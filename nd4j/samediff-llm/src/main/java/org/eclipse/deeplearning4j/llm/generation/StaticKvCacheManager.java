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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import org.nd4j.linalg.api.buffer.DataType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Static (dense) KV cache manager for CUDA graph-compatible decode.
 *
 * <p>Pre-allocates dense [batch, heads, maxKvLen, headDim] buffers at prefill time
 * and uses the native {@code kv_scatter} op (via {@link DecoderUtils#scatterNewKvEntries})
 * to write new entries at each decode step. The buffers have fixed shapes after
 * initialization, enabling CUDA graph capture and replay.</p>
 *
 * <h3>Scatter modes</h3>
 * <ul>
 *   <li><b>Java scatter</b> (default): Decoder outputs present KV tensors, Java calls
 *       the batched KvScatter op to copy entries into static buffers.</li>
 *   <li><b>C++ scatter</b>: When DSP's {@code configureKvCacheRetention} succeeds,
 *       the native plan handles KV scatter internally and only outputs logits.
 *       Controlled via the {@code kvScatterInCpp} flag.</li>
 * </ul>
 *
 * <h3>Input modes</h3>
 * <ul>
 *   <li><b>Padded mode</b> (DSP active): Full static buffer passed as input with
 *       sparse attention mask. Fixed shapes for CUDA graphs.</li>
 *   <li><b>View mode</b> (no DSP): KV views [0:cachePos] passed as input with
 *       all-ones attention mask. Growing shapes, no CUDA graph support.</li>
 * </ul>
 *
 * @see KvCacheManager
 * @see DecoderUtils#padKvCacheToStaticSize
 * @see DecoderUtils#scatterNewKvEntries
 */
@Slf4j
public class StaticKvCacheManager implements KvCacheManager {

    @Getter
    private Map<String, INDArray> staticKvBuffers;

    @Getter
    private long maxKvLen;

    @Getter
    private long cachePosition;

    private boolean initialized;

    /**
     * Whether C++ DSP handles KV scatter internally (set by caller after DSP configuration).
     */
    @Getter
    private boolean kvScatterInCpp;

    /** Model I/O configuration for name resolution. */
    private final ModelIOConfig ioConfig;

    public StaticKvCacheManager() {
        this(ModelIOConfig.builder().build());
    }

    public StaticKvCacheManager(ModelIOConfig ioConfig) {
        this.ioConfig = ioConfig;
        this.initialized = false;
        this.kvScatterInCpp = false;
        this.cachePosition = 0;
        this.maxKvLen = -1;
    }

    /**
     * Enable/disable C++ KV scatter mode.
     * When enabled, the decoder only outputs logits and C++ handles KV scatter.
     */
    public void setKvScatterInCpp(boolean enabled) {
        this.kvScatterInCpp = enabled;
    }

    @Override
    public KvCacheStrategy getStrategy() {
        return KvCacheStrategy.STATIC;
    }

    @Override
    public void initializeFromPrefill(Map<String, INDArray> prefillOutputs,
                                      DecoderUtils.KVCacheNames kvNames,
                                      int maxNewTokens, long prefillSeqLen) {
        this.maxKvLen = prefillSeqLen + maxNewTokens;
        log.info("  StaticKvCacheManager: prefillLen={}, maxKvLen={} ({} key + {} value tensors)",
                prefillSeqLen, maxKvLen, kvNames.keyNames.size(), kvNames.valueNames.size());

        this.staticKvBuffers = DecoderUtils.padKvCacheToStaticSize(
                prefillOutputs, kvNames.keyNames, kvNames.valueNames, maxKvLen);
        Nd4j.getExecutioner().commit();

        this.cachePosition = prefillSeqLen;
        this.initialized = true;
    }

    @Override
    public void initializeEmpty(SameDiff decoder, List<String> decoderInputNames,
                                int maxKvLen, long hiddenSize) {
        this.maxKvLen = maxKvLen;
        this.staticKvBuffers = new HashMap<>();

        List<String> pastInputNames = new ArrayList<>();
        for (String inputName : decoderInputNames) {
            if (ioConfig.isKvCacheInput(inputName)) {
                pastInputNames.add(inputName);
            }
        }

        for (String pastInputName : pastInputNames) {
            INDArray empty = DecoderUtils.createEmptyKvCache(decoder, pastInputName, 1, hiddenSize);
            long numHeads = empty.size(1);
            long headDim = empty.size(3);
            DataType kvType = empty.dataType();
            empty.close();

            staticKvBuffers.put(pastInputName,
                    Nd4j.zeros(kvType, 1, numHeads, maxKvLen, headDim));
        }

        this.cachePosition = 0;
        this.initialized = true;
        log.info("  StaticKvCacheManager: initialized empty with maxKvLen={}, {} buffers",
                maxKvLen, staticKvBuffers.size());
    }

    @Override
    public void prepareInputs(Map<String, INDArray> decoderInputMap, SameDiff decoder,
                              long hiddenSize, boolean dspActive) {
        if (!initialized || staticKvBuffers == null) return;

        for (String inputName : decoder.inputs()) {
            if (!ioConfig.isKvCacheInput(inputName)) continue;

            INDArray staticBuf = staticKvBuffers.get(inputName);
            if (staticBuf != null) {
                if (dspActive) {
                    // Padded mode: pass full static buffer (fixed shape for CUDA graphs)
                    decoderInputMap.put(inputName, staticBuf);
                } else {
                    // View mode: pass VIEW [0:cachePos] (growing shapes)
                    if (cachePosition > 0 && cachePosition < staticBuf.size(2)) {
                        INDArray view = staticBuf.get(
                                NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(0, cachePosition), NDArrayIndex.all());
                        decoderInputMap.put(inputName, view);
                    } else if (cachePosition >= staticBuf.size(2)) {
                        decoderInputMap.put(inputName, staticBuf);
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
        if (kvScatterInCpp) {
            // C++ handles scatter internally -- just advance position
            cachePosition++;
            return;
        }

        // Java scatter via batched native KvScatter op
        DecoderUtils.scatterNewKvEntries(staticKvBuffers, decoderOutputs,
                kvNames.keyNames, kvNames.valueNames, maxKvLen, cachePosition);
        cachePosition++;
    }

    @Override
    public void scatterMultipleEntries(Map<String, INDArray> decoderOutputs,
                                       DecoderUtils.KVCacheNames kvNames,
                                       int numEntries) {
        if (numEntries <= 0) return;

        DecoderUtils.scatterMultipleKvEntries(staticKvBuffers, decoderOutputs,
                kvNames.keyNames, kvNames.valueNames, maxKvLen, cachePosition, numEntries);
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
        return true;
    }

    @Override
    public Map<String, INDArray> getStaticKvBuffers() {
        return staticKvBuffers;
    }

    @Override
    public void close() {
        if (staticKvBuffers != null) {
            for (INDArray buf : staticKvBuffers.values()) {
                if (buf != null && !buf.wasClosed()) {
                    buf.setCloseable(true);
                    buf.close();
                }
            }
            staticKvBuffers = null;
        }
        initialized = false;
        log.info("StaticKvCacheManager closed");
    }
}
