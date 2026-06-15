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

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Greedy sampling strategy that always selects the highest probability token.
 *
 * <p>Greedy decoding is deterministic and produces the same output for the
 * same input. It tends to produce more coherent but potentially repetitive text.</p>
 *
 * <p>This is equivalent to argmax over the logits and is the fastest
 * sampling strategy.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public class GreedySampler implements Sampler {

    @Override
    public int sample(INDArray logits) {
        return SamplerUtils.argmax(logits);
    }

    @Override
    public int[] sampleBatch(INDArray logits) {
        return SamplerUtils.argmaxBatch(logits);
    }

    @Override
    public INDArray getDistribution(INDArray logits) {
        // Return softmax distribution for analysis, even though we just take argmax
        return SamplerUtils.softmax(logits);
    }

    @Override
    public String getName() {
        return "greedy";
    }
}
