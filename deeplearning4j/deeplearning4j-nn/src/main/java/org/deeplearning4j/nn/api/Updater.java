/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.deeplearning4j.nn.api;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.io.Serializable;

/**
 * Update the model
 *
 * @author Adam Gibson
 */
public interface Updater extends Serializable {

    /**
     * Set the internal (historical) state view array for this updater
     *
     * @param layer      Layer that this updater belongs to
     * @param viewArray  View array
     * @param initialize Whether to initialize the array or not
     */
    void setStateViewArray(Trainable layer, INDArray viewArray, boolean initialize);

    /**
     * @return the view array for this updater
     */
    INDArray getStateViewArray();

    /**
     * Updater: updates the model
     *
     * @param layer
     * @param gradient
     * @param iteration
     */
    void update(Trainable layer, Gradient gradient, int iteration, int epoch, int miniBatchSize, LayerWorkspaceMgr workspaceMgr);
}
