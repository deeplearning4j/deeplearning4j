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

package org.deeplearning4j.optimize.solvers.accumulation.encoding;

import java.io.Serializable;

/**
 * Used to combine ThresholdAlgorithm implementations in a way that is useful in a distributed training setting.
 * ThresholdAlgorithm instances can be combined to save their historical state (if any) that is used to make decisions
 * regarding the encoding threshold to use.
 * Without combining and storing the state, any threshold adaption would be lost between epochs.
 *
 * @author Alex Black
 */
public interface ThresholdAlgorithmReducer extends Serializable {

    /**
     * Add a ThresholdAlgorithm instance to the reducer
     * @param instance Instance to add. May be null.
     */
    void add(ThresholdAlgorithm instance);

    /**
     * Combine two reducers and return the result
     * @param other Other reducer to combine with this one
     * @return Combined reducer
     */
    ThresholdAlgorithmReducer merge(ThresholdAlgorithmReducer other);

    /**
     * @return The final threshold reducer after combining all instances
     */
    ThresholdAlgorithm getFinalResult();

}
