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

package org.deeplearning4j.rl4j.observation;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * The VoidObservation is a special kind of observation that the system will consider as an unavailable and undefined observation. <br>
 * Two examples of a transform issuing a VoidObservation is when an observation is skipped (SkippingObservation) or before the PoolingTransform has finished filling.
 *
 * @author Alexandre Boulanger
 */
public final class VoidObservation implements Observation {

    private static final VoidObservation instance = new VoidObservation();

    public static VoidObservation getInstance() {
        return instance;
    }

    private VoidObservation() {

    }

    @Override
    public INDArray toNDArray() {
        return null;
    }
}
