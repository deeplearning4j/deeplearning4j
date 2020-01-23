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

package org.deeplearning4j.rl4j.observation.prefiltering;

import org.deeplearning4j.rl4j.observation.channel.ChannelData;
import org.nd4j.base.Preconditions;

import java.util.List;

/**
 * A {@link PreFilter PreFilter} that will let pass 1 observation every _skipFrame_ observations.
 * With an exception: the last frame of an episode always pass.
 *
 * @author Alexandre Boulanger
 */
public class UniformSkippingPreFilter implements PreFilter {

    private final int skipFrame;

    /**
     * Creates a {@link PreFilter PreFilter} instance
     * @param skipFrame the PreFilter will let pass 1 observation every _skipFrame_ observations.
     */
    public UniformSkippingPreFilter(int skipFrame) {
        Preconditions.checkArgument(skipFrame > 0, "skipFrame should be greater than 0");

        this.skipFrame = skipFrame;
    }

    /**
     * @return always true if it's the final observation; true once every _skipFrame_ observations; otherwise false
     */
    public boolean isPassing(List<ChannelData> channelDataList, int currentObservationStep, boolean isFinalObservation) {
        return isFinalObservation || (currentObservationStep % skipFrame == 0);
    }
}
