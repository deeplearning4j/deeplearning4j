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

/**
 * A factory used to create {@link LegacyChannelData LegacyChannelData}
 *
 * @author Alexandre Boulanger
 */
public class LegacyChannelDataFactory {
    private final long[] observationShape;

    /**
     * Create an instance used to create {@link LegacyChannelData LegacyChannelData} instances
     * @param observationShape the target shape of the created LegacyChannelData's INDArray
     */
    public LegacyChannelDataFactory(long[] observationShape) {
        this.observationShape = observationShape;
    }

    /**
     * Create a {@link LegacyChannelData LegacyChannelData} instance with the factory's configured shape
     * @param input the data of the LegacyChannelData
     * @return the created LegacyChannelData
     */
    public ChannelData create(Encodable input) {
        return new LegacyChannelData(input, observationShape);
    }
}
