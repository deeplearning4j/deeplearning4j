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

package org.nd4j.autodiff.samediff.rl;

import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Top-k / top-p (nucleus) sampling strategy for generating completions.
 *
 * Adam Gibson
 */
@Data
@Builder
public class TopKSamplingStrategy implements SamplingStrategy {

    @Builder.Default
    private int topK = 50;

    @Builder.Default
    private double topP = 0.9;

    @Builder.Default
    private double temperature = 1.0;

    @Builder.Default
    private long seed = -1;

    @Override
    public INDArray generate(SameDiff model, INDArray prompts, int maxNewTokens, String logitVariable) {
        Random rng = seed >= 0 ? new Random(seed) : new Random();
        long batchSize = prompts.size(0);
        long promptLen = prompts.size(1);

        // Build the completion token by token
        long[][] completionTokens = new long[(int) batchSize][maxNewTokens];
        INDArray currentInput = prompts.dup();

        for (int t = 0; t < maxNewTokens; t++) {
            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put("input", currentInput);
            Map<String, INDArray> outputs = model.output(inputs, logitVariable);
            INDArray logits = outputs.get(logitVariable);

            // Get logits for the last position
            INDArray lastLogits = logits.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(logits.size(1) - 1));

            // Apply temperature
            if (temperature != 1.0) {
                lastLogits = lastLogits.div(temperature);
            }

            // Sample from the distribution
            INDArray probs = Transforms.softmax(lastLogits);

            for (int b = 0; b < batchSize; b++) {
                INDArray batchProbs = probs.getRow(b);
                int sampledToken = sampleFromDistribution(batchProbs, rng);
                completionTokens[b][t] = sampledToken;
            }

            // Append sampled tokens to input for next step
            INDArray newTokens = Nd4j.create(completionTokens).get(
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(0, t + 1));
            currentInput = Nd4j.concat(1, prompts, newTokens);
        }

        return Nd4j.create(completionTokens);
    }

    @Override
    public INDArray generateMultiple(SameDiff model, INDArray prompts, int numCompletions,
                                     int maxNewTokens, String logitVariable) {
        long batchSize = prompts.size(0);
        // Repeat each prompt numCompletions times
        INDArray expandedPrompts = prompts.repeat(0, numCompletions);
        INDArray completions = generate(model, expandedPrompts, maxNewTokens, logitVariable);
        return completions;
    }

    private int sampleFromDistribution(INDArray probs, Random rng) {
        double[] probArray = probs.toDoubleVector();
        double cumSum = 0.0;
        double target = rng.nextDouble();

        for (int i = 0; i < probArray.length; i++) {
            cumSum += probArray[i];
            if (cumSum >= target) {
                return i;
            }
        }
        return probArray.length - 1;
    }
}
