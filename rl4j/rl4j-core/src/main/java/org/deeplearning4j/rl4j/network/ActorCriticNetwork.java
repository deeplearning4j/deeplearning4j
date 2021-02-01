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
package org.deeplearning4j.rl4j.network;

import lombok.NonNull;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.common.base.Preconditions;
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

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private final NetworkHelper networkHelper = new NetworkHelper();

        private boolean isCombined;
        private ComputationGraph combinedNetwork;

        private ComputationGraph cgValueNetwork;
        private MultiLayerNetwork mlnValueNetwork;

        private ComputationGraph cgPolicyNetwork;
        private MultiLayerNetwork mlnPolicyNetwork;
        private String inputChannelName;
        private String[] channelNames;
        private ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] networkInputsToFeatureBindings;


        public Builder withCombinedNetwork(@NonNull ComputationGraph combinedNetwork) {
            isCombined = true;
            this.combinedNetwork = combinedNetwork;

            return this;
        }

        public Builder withSeparateNetworks(@NonNull ComputationGraph valueNetwork, @NonNull ComputationGraph policyNetwork) {
            this.cgValueNetwork = valueNetwork;
            this.cgPolicyNetwork = policyNetwork;
            isCombined = false;
            return this;
        }

        public Builder withSeparateNetworks(@NonNull MultiLayerNetwork valueNetwork, @NonNull ComputationGraph policyNetwork) {
            this.mlnValueNetwork = valueNetwork;
            this.cgPolicyNetwork = policyNetwork;
            isCombined = false;

            return this;
        }

        public Builder withSeparateNetworks(@NonNull ComputationGraph valueNetwork, @NonNull MultiLayerNetwork policyNetwork) {
            this.cgValueNetwork = valueNetwork;
            this.mlnPolicyNetwork = policyNetwork;
            isCombined = false;

            return this;
        }

        public Builder withSeparateNetworks(@NonNull MultiLayerNetwork valueNetwork, @NonNull MultiLayerNetwork policyNetwork) {
            this.mlnValueNetwork = valueNetwork;
            this.mlnPolicyNetwork = policyNetwork;
            isCombined = false;

            return this;
        }

        public Builder inputBindings(ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] networkInputsToFeatureBindings) {
            this.networkInputsToFeatureBindings = networkInputsToFeatureBindings;
            return this;
        }

        public Builder specificBinding(String inputChannelName) {
            this.inputChannelName = inputChannelName;
            return this;
        }

        public Builder channelNames(String[] channelNames) {
            this.channelNames = channelNames;
            return this;
        }

        public ActorCriticNetwork build() {
            INetworkHandler networkHandler;

            boolean isValueNetworkSet = !(cgValueNetwork == null && mlnValueNetwork == null);
            boolean isPolicyNetworkSet = !(cgPolicyNetwork == null && mlnPolicyNetwork == null);
            Preconditions.checkState(combinedNetwork != null || (isValueNetworkSet && isPolicyNetworkSet), "A network must be set.");

            if(isCombined) {
                networkHandler = (networkInputsToFeatureBindings == null)
                        ? networkHelper.buildHandler(combinedNetwork, inputChannelName, channelNames, LABEL_NAMES, CommonGradientNames.ActorCritic.Combined)
                        : networkHelper.buildHandler(combinedNetwork, networkInputsToFeatureBindings, channelNames, LABEL_NAMES, CommonGradientNames.ActorCritic.Combined);
            } else {
                INetworkHandler valueNetworkHandler;
                if(cgValueNetwork != null) {
                    valueNetworkHandler = (networkInputsToFeatureBindings == null)
                            ? networkHelper.buildHandler(cgValueNetwork, inputChannelName, channelNames, new String[] { CommonLabelNames.ActorCritic.Value }, CommonGradientNames.ActorCritic.Value)
                            : networkHelper.buildHandler(cgValueNetwork, networkInputsToFeatureBindings, channelNames, LABEL_NAMES, CommonGradientNames.ActorCritic.Value);
                } else {
                    valueNetworkHandler = networkHelper.buildHandler(mlnValueNetwork, inputChannelName, channelNames, CommonLabelNames.ActorCritic.Value, CommonGradientNames.ActorCritic.Value);
                }

                INetworkHandler policyNetworkHandler;
                if(cgPolicyNetwork != null) {
                    policyNetworkHandler = (networkInputsToFeatureBindings == null)
                            ? networkHelper.buildHandler(cgPolicyNetwork, inputChannelName, channelNames, new String[] { CommonLabelNames.ActorCritic.Policy }, CommonGradientNames.ActorCritic.Policy)
                            : networkHelper.buildHandler(cgPolicyNetwork, networkInputsToFeatureBindings, channelNames, LABEL_NAMES, CommonGradientNames.ActorCritic.Policy);
                } else {
                    policyNetworkHandler = networkHelper.buildHandler(mlnPolicyNetwork, inputChannelName, channelNames, CommonLabelNames.ActorCritic.Policy, CommonGradientNames.ActorCritic.Policy);
                }

                networkHandler = new CompoundNetworkHandler(valueNetworkHandler, policyNetworkHandler);
            }

            return new ActorCriticNetwork(networkHandler, isCombined);
        }

    }

}
