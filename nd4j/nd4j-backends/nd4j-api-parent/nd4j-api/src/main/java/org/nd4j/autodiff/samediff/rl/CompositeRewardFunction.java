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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Weighted combination of multiple reward functions.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public class CompositeRewardFunction implements RewardFunction {

    private final List<RewardFunction> functions = new ArrayList<>();
    private final List<Double> weights = new ArrayList<>();

    /**
     * Add a reward function with the given weight.
     */
    public CompositeRewardFunction add(RewardFunction function, double weight) {
        functions.add(function);
        weights.add(weight);
        return this;
    }

    @Override
    public INDArray score(INDArray prompts, INDArray completions) {
        if (functions.isEmpty()) {
            return Nd4j.zeros(prompts.size(0));
        }

        INDArray total = Nd4j.zeros(prompts.size(0));
        for (int i = 0; i < functions.size(); i++) {
            INDArray reward = functions.get(i).score(prompts, completions);
            total.addi(reward.mul(weights.get(i)));
        }
        return total;
    }
}
