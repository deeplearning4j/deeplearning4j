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
 * Used with the SignalingTransform. A listener that will be called when the transform's reset() or transform() is called.
 *
 * IMPORTANT NOTE: Do not store the Observation. Another transform may apply an operation that changes the
 * underlying INDArray (for example INDarray.muli()). Instead, use directly the observation in onTransform() or make a clone
 * (example by using clone = new SimpleObservation(input.toNDArray().add(0.0))
 *
 * @author Alexandre Boulanger
 */
public interface TransformListener {
    void onReset();

    /**
     * IMPORTANT NOTE: Do not store the Observation. Another transform may apply an operation that will changes the
     * underlying INDArray (for example INDarray.muli()). Instead, use directly the observation in onTransform() or make a clone
     * (example by using clone = new SimpleObservation(input.toNDArray().add(0.0))
     */
    void onTransform(Observation observation);
}