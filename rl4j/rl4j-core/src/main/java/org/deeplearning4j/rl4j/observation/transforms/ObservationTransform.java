/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.rl4j.observation.transforms;

import org.deeplearning4j.rl4j.observation.Observation;

/**
 * ObservationTransforms are used to transform the raw observations from the MDP into what policy and learning classes need. <br>
 *
 * @author Alexandre Boulanger
 */
public interface ObservationTransform {

    /**
     * Bring the ObservationTransform to an initial state.
     */
    void reset();

    /**
     * Transform the input observation. <br>
     * <br>
     * Outputting a VoidObservation will be interpreted as a skipped or unavailable observation.
     * For example, a stacking transform that returns the last X observations stacked together will
     * return VoidObservation until it has X elements in its pool.
     *
     * @param input The observation to be transformed
     */
    Observation transform(Observation input);

    /**
     * This indicates if the transform is 'primed' and ready to output observations. <br>
     * For example, a stacking transform that returns the last X observations stacked together will
     * return false until it has X elements in its pool.
     */
    boolean isReady();
}
