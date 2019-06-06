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

package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.stats.NormalizerStats;

import java.io.Serializable;

/**
 * Interface for strategies that can normalize and denormalize data arrays based on statistics of the population
 *
 * @author Ede Meijer
 */
public interface NormalizerStrategy<S extends NormalizerStats> extends Serializable {
    /**
     * Normalize a data array
     *
     * @param array the data to normalize
     * @param stats statistics of the data population
     */
    void preProcess(INDArray array, INDArray maskArray, S stats);

    /**
     * Denormalize a data array
     *
     * @param array the data to denormalize
     * @param stats statistics of the data population
     */
    void revert(INDArray array, INDArray maskArray, S stats);

    /**
     * Create a new {@link NormalizerStats.Builder} instance that can be used to fit new data and of the opType that
     * belongs to the current NormalizerStrategy implementation
     * 
     * @return the new builder
     */
    S.Builder newStatsBuilder();
}
