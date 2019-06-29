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

package org.deeplearning4j.rl4j.observation.pooling;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An ObservationPool is used with the PoolingTransform. This interface supervises how observations are stored
 * and what is returned to the PoolingTransform.transform() method.
 *
 * @author Alexandre Boulanger
 */
public interface ObservationPool {
    void add(INDArray observation);
    INDArray[] get();

    /**
     * Should return true when there are enough element to work with.
     */
    boolean isReady();
}
