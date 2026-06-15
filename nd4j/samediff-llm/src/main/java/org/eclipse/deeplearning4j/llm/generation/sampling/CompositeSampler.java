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

import lombok.Builder;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;

/**
 * Composite sampler that combines multiple sampling strategies.
 *
 * <p>This sampler applies transformations in order:</p>
 * <ol>
 *   <li>Temperature scaling (if temperature != 1.0)</li>
 *   <li>Top-K filtering (if topK > 0)</li>
 *   <li>Top-P (nucleus) filtering (if topP < 1.0)</li>
 *   <li>Softmax normalization</li>
 *   <li>Multinomial sampling</li>
 * </ol>
 *
 * <p>This is the most common sampling strategy used in practice,
 * allowing fine-grained control over the trade-off between
 * diversity and quality.</p>
 *
 * <p>Example:</p>
 * <pre>{@code
 * Sampler sampler = CompositeSampler.builder()
 *     .temperature(0.8)
 *     .topK(50)
 *     .topP(0.95)
 *     .build();
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Getter
public class CompositeSampler implements Sampler {

    private final double temperature;
    private final int topK;
    private final double topP;
    private final Random random;

    @Builder
    public CompositeSampler(double temperature, int topK, double topP, Long seed) {
        this.temperature = temperature > 0 ? temperature : 1.0;
        this.topK = topK > 0 ? topK : 0;
        this.topP = (topP > 0 && topP < 1.0) ? topP : 1.0;
        this.random = seed != null ? new Random(seed) : new Random();
    }

    @Override
    public int sample(INDArray logits) {
        INDArray probs = getDistribution(logits);
        return SamplerUtils.multinomialSample(probs, random);
    }

    @Override
    public int[] sampleBatch(INDArray logits) {
        INDArray probs = getDistribution(logits);
        return SamplerUtils.multinomialSampleBatch(probs, random);
    }

    @Override
    public INDArray getDistribution(INDArray logits) {
        INDArray processed = logits;

        // 1. Apply temperature scaling
        if (temperature != 1.0) {
            processed = SamplerUtils.applyTemperature(processed, temperature);
        }

        // 2. Apply top-K filtering
        if (topK > 0) {
            processed = SamplerUtils.topKFilter(processed, topK);
        }

        // 3. Apply top-P filtering
        if (topP < 1.0) {
            processed = SamplerUtils.topPFilter(processed, topP);
        }

        // 4. Convert to probabilities
        return SamplerUtils.softmax(processed);
    }

    @Override
    public String getName() {
        StringBuilder name = new StringBuilder("composite(");
        if (temperature != 1.0) {
            name.append("temp=").append(temperature);
        }
        if (topK > 0) {
            if (name.length() > 10) name.append(",");
            name.append("topK=").append(topK);
        }
        if (topP < 1.0) {
            if (name.length() > 10) name.append(",");
            name.append("topP=").append(topP);
        }
        name.append(")");
        return name.toString();
    }

    /**
     * Create a sampler with only temperature scaling.
     *
     * @param temperature the temperature value
     * @return temperature-only sampler
     */
    public static CompositeSampler withTemperature(double temperature) {
        return CompositeSampler.builder()
                .temperature(temperature)
                .build();
    }

    /**
     * Create a sampler with only top-K filtering.
     *
     * @param k number of top tokens to consider
     * @return top-K sampler
     */
    public static CompositeSampler withTopK(int k) {
        return CompositeSampler.builder()
                .topK(k)
                .build();
    }

    /**
     * Create a sampler with only top-P (nucleus) filtering.
     *
     * @param p cumulative probability threshold
     * @return top-P sampler
     */
    public static CompositeSampler withTopP(double p) {
        return CompositeSampler.builder()
                .topP(p)
                .build();
    }
}
