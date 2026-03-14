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

import lombok.Getter;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

/**
 * Reward function backed by a SameDiff reward model.
 * Runs a forward pass through the reward model and extracts the reward output.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public class SameDiffRewardFunction implements RewardFunction {

    @Getter
    private final SameDiff rewardModel;

    @Getter
    private final String inputVariable;

    @Getter
    private final String rewardOutputVariable;

    /**
     * @param rewardModel          The SameDiff reward model
     * @param inputVariable        Name of the input variable (token IDs)
     * @param rewardOutputVariable Name of the output variable (scalar rewards)
     */
    public SameDiffRewardFunction(SameDiff rewardModel, String inputVariable, String rewardOutputVariable) {
        this.rewardModel = rewardModel;
        this.inputVariable = inputVariable;
        this.rewardOutputVariable = rewardOutputVariable;
    }

    @Override
    public INDArray score(INDArray prompts, INDArray completions) {
        // Concatenate prompts and completions along sequence dimension
        INDArray fullSequence = org.nd4j.linalg.factory.Nd4j.concat(1, prompts, completions);

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put(inputVariable, fullSequence);

        Map<String, INDArray> outputs = rewardModel.output(inputs, rewardOutputVariable);
        return outputs.get(rewardOutputVariable);
    }
}
