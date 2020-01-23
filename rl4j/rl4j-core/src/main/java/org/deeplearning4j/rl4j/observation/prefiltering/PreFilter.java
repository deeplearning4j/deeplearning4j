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

import java.util.List;

/**
 * Used by {@link org.deeplearning4j.rl4j.observation.ObservationHandler ObservationHandler} to determine if
 * it should consider the observation as skipped.
 *
 * @author Alexandre Boulanger
 */
public interface PreFilter {
    /**
     * Determines if the observation should be skipped.
     * @param channelDataList A list of channel data for the current observation
     * @param currentObservationStep The step, relative to the start of the episode, of the current observation
     * @param isFinalObservation true if it's the last observation of the episode
     * @return true if it passes the filter; false if the observation should be skipped
     */
    boolean isPassing(List<ChannelData> channelDataList, int currentObservationStep, boolean isFinalObservation);
}
