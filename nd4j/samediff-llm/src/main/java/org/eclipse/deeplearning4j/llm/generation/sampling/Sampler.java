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

package org.eclipse.deeplearning4j.llm.generation.sampling;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Interface for token sampling strategies in text generation.
 *
 * <p>Samplers select the next token from a probability distribution
 * over the vocabulary. Different strategies provide trade-offs between
 * determinism and diversity in generated text.</p>
 *
 * <p>Implementations should be thread-safe for use in parallel generation.</p>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * Sampler sampler = Sampler.fromConfig(config);
 * INDArray logits = model.getLogits();
 * int nextToken = sampler.sample(logits);
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public interface Sampler {

    /**
     * Sample the next token from logits.
     *
     * @param logits the raw logits from the model, shape [vocabSize] or [1, vocabSize]
     * @return the sampled token ID
     */
    int sample(INDArray logits);

    /**
     * Sample the next token from logits for a batch.
     *
     * @param logits the raw logits, shape [batchSize, vocabSize]
     * @return array of sampled token IDs, one per batch element
     */
    int[] sampleBatch(INDArray logits);

    /**
     * Get the probability distribution after applying this sampler's transformations.
     *
     * <p>This is useful for logging, analysis, or beam search implementations
     * that need access to the modified distribution.</p>
     *
     * @param logits the raw logits
     * @return the probability distribution after transformations
     */
    INDArray getDistribution(INDArray logits);

    /**
     * Get the name of this sampling strategy.
     *
     * @return human-readable name
     */
    String getName();

    /**
     * Create a sampler from configuration.
     *
     * <p>This factory method creates an appropriate sampler based on the
     * configuration parameters. It may return a composite sampler that
     * combines multiple strategies (e.g., temperature + top-k + top-p).</p>
     *
     * @param config the sampling configuration
     * @return appropriate sampler implementation
     */
    static Sampler fromConfig(SamplingConfig config) {
        if (config.isGreedy()) {
            return new GreedySampler();
        }

        // Build a composite sampler with the configured strategies
        return CompositeSampler.builder()
                .temperature(config.getTemperature())
                .topK(config.getTopK())
                .topP(config.getTopP())
                .seed(config.getSeed())
                .build();
    }
}
