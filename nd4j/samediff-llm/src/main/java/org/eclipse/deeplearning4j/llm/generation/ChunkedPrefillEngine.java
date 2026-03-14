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
 * Chunked/windowed prefill engine for processing long prompts.
 *
 * <p>Long prompts can exceed GPU memory when processed in a single forward pass due to
 * the quadratic memory cost of self-attention. This engine splits the prompt embeddings
 * into fixed-size chunks and processes each chunk sequentially, building up the KV cache
 * incrementally.</p>
 *
 * <h3>Algorithm</h3>
 * <ol>
 *   <li>Split input embeddings into chunks of {@code chunkSize} tokens</li>
 *   <li>For each chunk:
 *     <ul>
 *       <li>Build a causal mask covering [chunkSize, pastLen + chunkSize]</li>
 *       <li>Run the model forward pass with the chunk embeddings and accumulated KV cache</li>
 *       <li>Concatenate new KV outputs with existing cache</li>
 *     </ul>
 *   </li>
 *   <li>Return the logits from the final chunk</li>
 * </ol>
 *
 * <p>This trades off prefill latency (more forward passes) for bounded peak memory usage.
 * A chunk size of 512 is a good default for most models on consumer GPUs.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see DecoderUtils
 */
@Slf4j
public class ChunkedPrefillEngine {

    /** Default chunk size in tokens. */
    public static final int DEFAULT_CHUNK_SIZE = 512;

    @Getter private final SameDiff model;
    @Getter private final int chunkSize;

    /**
     * Create a chunked prefill engine.
     *
     * @param model     the SameDiff decoder model
     * @param chunkSize number of tokens per chunk (default: 512)
     */
    public ChunkedPrefillEngine(SameDiff model, int chunkSize) {
        if (model == null) throw new IllegalArgumentException("model must not be null");
        if (chunkSize < 1) throw new IllegalArgumentException("chunkSize must be >= 1, got " + chunkSize);

        this.model = model;
        this.chunkSize = chunkSize;
    }

    /**
     * Create a chunked prefill engine with the default chunk size of 512.
     *
     * @param model the SameDiff decoder model
     */
    public ChunkedPrefillEngine(SameDiff model) {
        this(model, DEFAULT_CHUNK_SIZE);
    }

    /**
     * Process a full prompt through chunked prefill.
     *
     * <p>Splits the embeddings into chunks, processes each one sequentially through
     * the model, and accumulates KV cache entries. Returns the final logits from
     * the last chunk.</p>
     *
     * @param embeddings full prompt embeddings, shape [1, seqLen, hiddenSize]
     * @param kvCaches   mutable map of KV cache input names to their tensors;
     *                   updated in-place with accumulated KV entries after each chunk
     * @return logits from the final chunk, shape [1, chunkSize, vocabSize] or
     *         [1, remainderSize, vocabSize] for the last chunk
     */
    public INDArray prefill(INDArray embeddings, Map<String, INDArray> kvCaches) {
        long totalLen = embeddings.size(1);
        int numChunks = (int) Math.ceil((double) totalLen / chunkSize);
        long pastLen = 0;
        INDArray lastLogits = null;

        log.info("Chunked prefill: {} tokens in {} chunks of {} (last chunk: {} tokens)",
                totalLen, numChunks, chunkSize,
                totalLen - (long)(numChunks - 1) * chunkSize);

        for (int c = 0; c < numChunks; c++) {
            long chunkStart = (long) c * chunkSize;
            long chunkEnd = Math.min(chunkStart + chunkSize, totalLen);
            int currentChunkSize = (int) (chunkEnd - chunkStart);

            // Slice this chunk's embeddings: [1, currentChunkSize, hiddenSize]
            INDArray chunkEmbeddings = embeddings.get(
                    NDArrayIndex.all(),
                    NDArrayIndex.interval(chunkStart, chunkEnd),
                    NDArrayIndex.all());

            log.debug("Processing chunk {}/{}: tokens [{}, {}), pastLen={}",
                    c + 1, numChunks, chunkStart, chunkEnd, pastLen);

            lastLogits = prefillOneChunk(chunkEmbeddings, pastLen, kvCaches);
            pastLen += currentChunkSize;
        }

        return lastLogits;
    }

    /**
     * Process a single chunk through the model.
     *
     * <p>Builds a causal mask for the chunk, runs the model forward pass, and
     * concatenates the new KV entries with the existing cache.</p>
     *
     * @param chunkEmbeddings embeddings for this chunk, shape [1, chunkLen, hiddenSize]
     * @param pastLen         number of tokens already processed (KV cache length)
     * @param kvCaches        mutable map of KV cache names to tensors; updated in-place
     * @return logits for this chunk, shape [1, chunkLen, vocabSize]
     */
    public INDArray prefillOneChunk(INDArray chunkEmbeddings, long pastLen,
                                     Map<String, INDArray> kvCaches) {
        long chunkLen = chunkEmbeddings.size(1);
        long totalLen = pastLen + chunkLen;

        // Build causal mask: [1, 1, chunkLen, totalLen]
        INDArray causalMask = DecoderUtils.buildCausalMask(chunkLen, totalLen);

        // Build input map for the model
        Map<String, INDArray> inputMap = new HashMap<>();
        inputMap.put("inputs_embeds", chunkEmbeddings);
        inputMap.put("_causal_mask", causalMask);

        // Build position IDs for this chunk
        INDArray positionIds = Nd4j.arange(pastLen, pastLen + chunkLen)
                .reshape(1, chunkLen)
                .castTo(DataType.LONG);
        inputMap.put("position_ids", positionIds);

        // Build attention mask: all ones for [1, totalLen]
        INDArray attentionMask = Nd4j.ones(DataType.LONG, 1, totalLen);
        inputMap.put("attention_mask", attentionMask);

        // Add current KV cache entries
        inputMap.putAll(kvCaches);

        // Find output names
        String logitsName = DecoderUtils.findLogitsOutputName(model);
        DecoderUtils.KVCacheNames kvNames = DecoderUtils.findKVCacheOutputNames(model);

        // Collect all output names we need
        List<String> outputNames = new ArrayList<>();
        if (logitsName != null) outputNames.add(logitsName);
        outputNames.addAll(kvNames.keyNames);
        outputNames.addAll(kvNames.valueNames);

        // Run forward pass
        Map<String, INDArray> outputs = model.output(inputMap, outputNames);

        // Update KV caches with the new present values
        for (String keyName : kvNames.keyNames) {
            INDArray presentKey = outputs.get(keyName);
            if (presentKey != null) {
                String pastName = keyName.replace("present", "past_key_values");
                kvCaches.put(pastName, presentKey);
            }
        }
        for (String valueName : kvNames.valueNames) {
            INDArray presentValue = outputs.get(valueName);
            if (presentValue != null) {
                String pastName = valueName.replace("present", "past_key_values");
                kvCaches.put(pastName, presentValue);
            }
        }

        // Clean up temporaries
        causalMask.close();
        positionIds.close();
        attentionMask.close();

        return logitsName != null ? outputs.get(logitsName) : null;
    }

    @Override
    public String toString() {
        return String.format("ChunkedPrefillEngine[chunkSize=%d, model=%s]",
                chunkSize, model.toString());
    }
}
