/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.deeplearning4j.rl4j.network;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An Actor-Critic network implementation <br/>
 * Label names: "value" and "policy" <br/>
 * <br/>
 * Gradient names:
 * <ul>
 *     <li>A <b>single</b> network for the value and policy: "combined"</li>
 *     <li>A <b>separate</b> network for the value and policy: "value" and "policy"</li>
 * </ul>
 */
public class ActorCriticNetwork extends BaseNetwork<ActorCriticNetwork> {

    private static final String[] LABEL_NAMES = new String[] {
            CommonLabelNames.ActorCritic.Value,
            CommonLabelNames.ActorCritic.Policy
    };

    private final boolean isCombined;

    public ActorCriticNetwork(ComputationGraph combinedNetwork) {
        this(new ComputationGraphHandler(combinedNetwork, LABEL_NAMES, CommonGradientNames.ActorCritic.Combined), true);
    }

    public ActorCriticNetwork(ComputationGraph valueNetwork, ComputationGraph policyNetwork) {
        this(createValueNetworkHandler(valueNetwork), createPolicyNetworkHandler(policyNetwork));
    }

    public ActorCriticNetwork(MultiLayerNetwork valueNetwork, ComputationGraph policyNetwork) {
        this(createValueNetworkHandler(valueNetwork), createPolicyNetworkHandler(policyNetwork));
    }

    public ActorCriticNetwork(ComputationGraph valueNetwork, MultiLayerNetwork policyNetwork) {
        this(createValueNetworkHandler(valueNetwork), createPolicyNetworkHandler(policyNetwork));
    }

    public ActorCriticNetwork(MultiLayerNetwork valueNetwork, MultiLayerNetwork policyNetwork) {
        this(createValueNetworkHandler(valueNetwork), createPolicyNetworkHandler(policyNetwork));
    }

    private static INetworkHandler createValueNetworkHandler(ComputationGraph valueNetwork) {
        return new ComputationGraphHandler(valueNetwork, new String[] { CommonLabelNames.ActorCritic.Value }, CommonGradientNames.ActorCritic.Value);
    }

    private static INetworkHandler createValueNetworkHandler(MultiLayerNetwork valueNetwork) {
        return new MultiLayerNetworkHandler(valueNetwork, CommonLabelNames.ActorCritic.Value, CommonGradientNames.ActorCritic.Value);
    }

    private static INetworkHandler createPolicyNetworkHandler(ComputationGraph policyNetwork) {
        return new ComputationGraphHandler(policyNetwork, new String[] { CommonLabelNames.ActorCritic.Policy }, CommonGradientNames.ActorCritic.Policy);
    }

    private static INetworkHandler createPolicyNetworkHandler(MultiLayerNetwork policyNetwork) {
        return new MultiLayerNetworkHandler(policyNetwork, CommonLabelNames.ActorCritic.Policy, CommonGradientNames.ActorCritic.Policy);
    }

    private ActorCriticNetwork(INetworkHandler valueNetworkHandler, INetworkHandler policyNetworkHandler) {
        this(new CompoundNetworkHandler(valueNetworkHandler, policyNetworkHandler), false);
    }

    private ActorCriticNetwork(INetworkHandler handler, boolean isCombined) {
        super(handler);
        this.isCombined = isCombined;
    }

    @Override
    protected NeuralNetOutput packageResult(INDArray[] output) {
        NeuralNetOutput result = new NeuralNetOutput();
        result.put(CommonOutputNames.ActorCritic.Value, output[0]);
        result.put(CommonOutputNames.ActorCritic.Policy, output[1]);

        return result;
    }

    @Override
    public ActorCriticNetwork clone() {
        return new ActorCriticNetwork(getNetworkHandler().clone(), isCombined);
    }
}
