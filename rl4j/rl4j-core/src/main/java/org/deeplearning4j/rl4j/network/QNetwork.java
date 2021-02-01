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
 * A QNetwork implementation.<br/>
 * Label names: "Q"<br/>
 * Gradient names: "Q"<br/>
 */
public class QNetwork extends BaseNetwork<QNetwork> {
    private static final String[] LABEL_NAMES = new String[] {
            CommonLabelNames.QValues
    };

    private QNetwork(INetworkHandler handler) {
        super(handler);
    }

    @Override
    protected NeuralNetOutput packageResult(INDArray[] output) {
        NeuralNetOutput result = new NeuralNetOutput();
        result.put(CommonOutputNames.QValues, output[0]);

        return result;
    }

    @Override
    public QNetwork clone() {
        return new QNetwork(getNetworkHandler().clone());
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private final NetworkHelper networkHelper = new NetworkHelper();

        private ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] networkInputsToFeatureBindings;
        private String[] channelNames;
        private String inputChannelName;

        private ComputationGraph cgNetwork;
        private MultiLayerNetwork mlnNetwork;

        public Builder withNetwork(@NonNull ComputationGraph network) {
            this.cgNetwork = network;
            return this;
        }

        public Builder withNetwork(@NonNull MultiLayerNetwork network) {
            this.mlnNetwork = network;
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

        public QNetwork build() {
            INetworkHandler networkHandler;

            Preconditions.checkState(cgNetwork != null || mlnNetwork != null, "A network must be set.");

            if(cgNetwork != null) {
                networkHandler = (networkInputsToFeatureBindings == null)
                        ? networkHelper.buildHandler(cgNetwork, inputChannelName, channelNames, LABEL_NAMES, CommonGradientNames.QValues)
                        : networkHelper.buildHandler(cgNetwork, networkInputsToFeatureBindings, channelNames, LABEL_NAMES, CommonGradientNames.QValues);
            } else {
                networkHandler = networkHelper.buildHandler(mlnNetwork, inputChannelName, channelNames, CommonLabelNames.QValues, CommonGradientNames.QValues);
            }

            return new QNetwork(networkHandler);
        }
    }

}