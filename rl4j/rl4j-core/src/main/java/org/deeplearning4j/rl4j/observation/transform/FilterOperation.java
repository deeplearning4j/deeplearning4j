/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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
package org.deeplearning4j.rl4j.observation.transform;

import java.util.Map;

/**
 * Used with {@link TransformProcess TransformProcess} to filter-out an observation.
 *
 * @author Alexandre Boulanger
 */
public interface FilterOperation {
    /**
     * The logic that determines if the observation should be skipped.
     *
     * @param channelsData the name of the channel
     * @param currentObservationStep The step number if the observation in the current episode.
     * @param isFinalObservation true if this is the last observation of the episode
     * @return true if the observation should be skipped
     */
    boolean isSkipped(Map<String, Object> channelsData, int currentObservationStep, boolean isFinalObservation);
}
