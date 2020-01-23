/*******************************************************************************
 * Copyright (c) 2020 Skymind, Inc.
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

package org.deeplearning4j.rl4j.observation.channel.legacy;

import org.deeplearning4j.rl4j.observation.channel.ChannelData;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A {@link ChannelData ChannelData} used with legacy MDPs (those that uses Encodable)
 *
 * @author Alexandre Boulanger
 */
public class LegacyChannelData implements ChannelData {

    private final INDArray data;

    /**
     * Creates a LegacyChannelData
     * @param input An Encodable that contains the data
     * @param observationShape The target shape of the INDArray
     */
    public LegacyChannelData(Encodable input, long[] observationShape) {
        data = getInput(input, observationShape);
    }

    @Override
    public INDArray toINDArray() {
        return data;
    }

    protected static INDArray getInput(Encodable obs, long[] observationShape) {
        INDArray arr = Nd4j.create(obs.toArray());
        if (observationShape.length == 1)
            return arr.reshape(new long[] {1, arr.length()});
        else
            return arr.reshape(observationShape);
    }
}
