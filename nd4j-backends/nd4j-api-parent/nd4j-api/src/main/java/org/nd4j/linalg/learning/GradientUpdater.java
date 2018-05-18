/*-
 *
 *  * Copyright 2017 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */
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
