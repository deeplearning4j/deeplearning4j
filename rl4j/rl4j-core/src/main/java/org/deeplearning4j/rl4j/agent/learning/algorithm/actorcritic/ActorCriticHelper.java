/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.rl4j.agent.learning.algorithm.actorcritic;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A base helper class for the Actor Critic update algorithm. The algorithm is the same whether it's used with a RNN or
 * not but, the shape of INDArrays are different. This class, {@link NonRecurrentActorCriticHelper},
 * and {@link RecurrentActorCriticHelper} handle the differences.
 */
public abstract class ActorCriticHelper {
    /**
     * Create an empty INDArray to be used as the value array
     * @param trainingBatchSize the size of the training batch
     * @return An empty value array
     */
    public abstract INDArray createValueLabels(int trainingBatchSize);

    /**
     * Create an empty INDArray to be used as the policy array
     * @param trainingBatchSize the size of the training batch
     * @return An empty policy array
     */
    public abstract INDArray createPolicyLabels(int trainingBatchSize);

    /**
     * Set the advantage for a given action and training batch index in the policy array
     * @param policy The policy array
     * @param idx The training batch index
     * @param action The action
     * @param advantage The advantage value
     */
    public abstract void setPolicy(INDArray policy, long idx, int action, double advantage);
}
