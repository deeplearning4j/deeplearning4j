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

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NonNull;
import org.apache.commons.collections4.map.HashedMap;
import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * A class that maps the channels of an {@link Observation} or a {@link Features} to the inputs of a network.
 */
public class ChannelToNetworkInputMapper {
    private final IdxBinding[] networkInputsToChannelNameMap;
    private final int inputCount;

    /**
     * @param networkInputsToChannelNameMap An array that describe how to map the network inputs with the channel names.
     * @param networkInputNames An ordered array of the network inputs.
     * @param channelNames An ordered array of the observation/features channel names
     */
    public ChannelToNetworkInputMapper(@NonNull NetworkInputToChannelBinding[] networkInputsToChannelNameMap,
                                       String[] networkInputNames,
                                       String[] channelNames) {
        Preconditions.checkArgument(networkInputsToChannelNameMap.length > 0, "networkInputsToChannelNameMap is empty.");
        Preconditions.checkArgument(networkInputNames.length > 0, "networkInputNames is empty.");
        Preconditions.checkArgument(channelNames.length > 0, "channelNames is empty.");

        // All network inputs must be mapped exactly once.
        for (String inputName : networkInputNames) {
            int numTimesMapped = 0;
            for (NetworkInputToChannelBinding networkInputToChannelBinding : networkInputsToChannelNameMap) {
                numTimesMapped += inputName == networkInputToChannelBinding.networkInputName ? 1 : 0;
            }

            if (numTimesMapped != 1) {
                throw new IllegalArgumentException("All network inputs must be mapped exactly once. Input '" + inputName + "' is mapped " + numTimesMapped + " times.");
            }
        }

        Map<String, Integer> networkNameToIdx = new HashedMap<String, Integer>();
        for(int i = 0; i < networkInputNames.length; ++i) {
            networkNameToIdx.put(networkInputNames[i], i);
        }

        Map<String, Integer> channelNamesToIdx = new HashedMap<String, Integer>();
        for(int i = 0; i < channelNames.length; ++i) {
            channelNamesToIdx.put(channelNames[i], i);
        }

        this.networkInputsToChannelNameMap = new IdxBinding[networkInputNames.length];
        for(int i = 0; i < networkInputsToChannelNameMap.length; ++i) {
            NetworkInputToChannelBinding nameMap = networkInputsToChannelNameMap[i];

            Integer networkIdx = networkNameToIdx.get(nameMap.networkInputName);
            if(networkIdx == null) {
                throw new IllegalArgumentException("'" + nameMap.networkInputName + "' not found in networkInputNames");
            }

            Integer channelIdx = channelNamesToIdx.get(nameMap.channelName);
            if(channelIdx == null) {
                throw new IllegalArgumentException("'" + nameMap.channelName + "' not found in channelNames");
            }

            this.networkInputsToChannelNameMap[i]  = new IdxBinding(networkIdx, channelIdx);
        }

        inputCount = networkInputNames.length;
    }

    public INDArray[] getNetworkInputs(Observation observation) {
        INDArray[] result = new INDArray[inputCount];
        for(IdxBinding map : networkInputsToChannelNameMap) {
            result[map.networkInputIdx] = observation.getChannelData(map.channelIdx);
        }

        return result;
    }

    public INDArray[] getNetworkInputs(Features features) {
        INDArray[] result = new INDArray[inputCount];
        for(IdxBinding map : networkInputsToChannelNameMap) {
            result[map.networkInputIdx] = features.get(map.channelIdx);
        }

        return result;
    }


    @AllArgsConstructor
    public static class NetworkInputToChannelBinding {
        @Getter
        private String networkInputName;
        @Getter
        private String channelName;

        public static NetworkInputToChannelBinding map(String networkInputName, String channelName) {
            return new NetworkInputToChannelBinding(networkInputName, channelName);
        }
    }

    @AllArgsConstructor
    private static class IdxBinding {
        int networkInputIdx;
        int channelIdx;
    }

}
