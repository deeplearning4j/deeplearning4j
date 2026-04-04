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

package org.eclipse.deeplearning4j.audio.whisper;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.eclipse.deeplearning4j.llm.generation.StaticKvCacheDecodeLoop;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashSet;
import java.util.Set;

/**
 * Whisper decoder that delegates to {@link StaticKvCacheDecodeLoop} with encoder-decoder support.
 * <p>
 * All decode paths use the samediff-llm infrastructure for KV cache management,
 * GPU-side token sampling, and optional CUDA graph replay.
 */
@Slf4j
public class WhisperDecoder {

    private int maxTokens = 448;
    private KvCacheStrategy kvCacheStrategy = KvCacheStrategy.STATIC;
    private SamplingConfig samplingConfig = SamplingConfig.greedy();

    @Builder
    public WhisperDecoder(int maxTokens, KvCacheStrategy kvCacheStrategy, SamplingConfig samplingConfig) {
        this.maxTokens = maxTokens > 0 ? maxTokens : 448;
        this.kvCacheStrategy = kvCacheStrategy != null ? kvCacheStrategy : KvCacheStrategy.STATIC;
        this.samplingConfig = samplingConfig != null ? samplingConfig : SamplingConfig.greedy();
    }

    /**
     * Decode using {@link StaticKvCacheDecodeLoop} with encoder-decoder mode.
     * <p>
     * The encoder output is passed to the decoder at every step as cross-attention context.
     *
     * @param decoder        The decoder SameDiff graph
     * @param tokenizer      The Whisper tokenizer
     * @param encoderOutput  Encoder output of shape [1, seqLen, hiddenSize]
     * @param promptTokens   Initial prompt tokens (SOT + language + task)
     * @return GenerationResult with decoded tokens and timing metrics
     */
    public GenerationResult decode(SameDiff decoder, WhisperTokenizer tokenizer,
                                    INDArray encoderOutput, int[] promptTokens) {
        Set<Integer> stopTokens = new HashSet<>();
        stopTokens.add(WhisperTokenizer.EOT);

        // Build dummy prefill embeddings for shape inference
        long hiddenSize = encoderOutput.size(2);
        INDArray prefillEmbeddings = Nd4j.zeros(DataType.FLOAT, 1, promptTokens.length, hiddenSize);

        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(decoder)
                .tokenizer(tokenizer)
                .samplingConfig(samplingConfig)
                .maxNewTokens(maxTokens)
                .kvCacheStrategy(kvCacheStrategy)
                .additionalStopTokenIds(stopTokens)
                .hiddenSize(hiddenSize)
                .encoderDecoder(true)
                .encoderOutputs(encoderOutput)
                .ioConfig(ModelIOConfig.builder()
                        .encoderDecoder(true)
                        .build())
                .build();

        return loop.decode(prefillEmbeddings, promptTokens);
    }
}
