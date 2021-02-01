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
package org.deeplearning4j.rl4j.observation.transform.operation.historymerge;

import org.deeplearning4j.rl4j.observation.transform.operation.HistoryMergeTransform;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * HistoryMergeElementStore is used with the {@link HistoryMergeTransform HistoryMergeTransform}. Used to supervise how data from the
 * HistoryMergeTransform is stored.
 *
 * @author Alexandre Boulanger
 */
public interface HistoryMergeElementStore {
    /**
     * Add an element into the store
     * @param observation
     */
    void add(INDArray observation);

    /**
     * Get the content of the store
     * @return the content of the store
     */
    INDArray[] get();

    /**
     * Used to tell the HistoryMergeTransform that the store is ready. The HistoryMergeTransform will tell the {@link org.deeplearning4j.rl4j.observation.transform.TransformProcess TransformProcess}
     * to skip the observation is the store is not ready.
     * @return true if the store is ready
     */
    boolean isReady();

    /**
     * Resets the store to an initial state.
     */
    void reset();
}
