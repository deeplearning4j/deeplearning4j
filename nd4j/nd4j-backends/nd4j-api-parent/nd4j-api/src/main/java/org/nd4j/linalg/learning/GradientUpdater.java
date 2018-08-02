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

package org.nd4j.linalg.learning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;

/**
 * Gradient modifications: Calculates an update and tracks related information for gradient changes over time
 * for handling updates.
 *
 * @author Alex Black
 */
public interface GradientUpdater<T extends IUpdater> {

    T getConfig();

    /**
     * For the internal updater state (if any): set this to use the provided array.
     * Used during initialization, and when restoring the updater state (after serialization, for example)
     *  @param viewArray    Array (that is a view of a larger array) to use for the state.
     * @param gradientShape
     * @param gradientOrder
     * @param initialize   If true: the updater must initialize the view array. If false: no change to view array contents
     */
    void setStateViewArray(INDArray viewArray, long[] gradientShape, char gradientOrder, boolean initialize);

    /**
     * Modify the gradient to be an update. Note that this is be done in-place
     *
     * @param gradient  the gradient to modify
     * @param iteration
     * @return the modified gradient
     */
    void applyUpdater(INDArray gradient, int iteration, int epoch);
}
