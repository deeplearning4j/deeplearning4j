/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
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

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * ThresholdAlgorithm is responsible for determining the threshold to use when encoding updates for distributed training.
 * It is used to implement adaptive threshold encoding approaches such as {@link org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold.AdaptiveThresholdAlgorithm}
 *
 * @author Alex Black
 */
public interface ThresholdAlgorithm extends Serializable {

    /**
     *
     * @param iteration           Current neural network training iteration
     * @param epoch               Current neural network training epoch
     * @param lastThreshold       The encoding threshold used in the last iteration - if available. May be null for first
     *                            iteration in an epoch (where no 'last iteration' value is available)
     * @param lastWasDense        Whether the last encoding was dense (true) or sparse (false). May be null for the first
     *                            iteration in an epoch (where no 'last iteration' value is available)
     * @param lastSparsityRatio   The sparsity ratio of the last iteration. Sparsity ratio is defined as
     *                            numElements(encoded)/length(updates). A sparsity ratio of 1.0 would mean all entries
     *                            present in encoded representation; a sparsity ratio of 0.0 would mean the encoded vector
     *                            did not contain any values.
     *                            Note: when the last encoding was dense, lastSparsityRatio is always null - this means
     *                            that the sparsity ratio is larger than 1/16 = 0.0625
     * @param updatesPlusResidual The actual array (updates plus residual) that will be encoded using the threshold
     *                            calculated/returned by this method
     * @return
     */
    double calculateThreshold(int iteration, int epoch, Double lastThreshold, Boolean lastWasDense, Double lastSparsityRatio,
                              INDArray updatesPlusResidual);

    /**
     * Create a new ThresholdAlgorithmReducer.
     * Note that implementations should NOT add the curret ThresholdAlgorithm to it.
     *
     * @return ThresholdAlgorithmReducer
     */
    ThresholdAlgorithmReducer newReducer();

    /**
     * @return A clone of the current threshold algorithm
     */
    ThresholdAlgorithm clone();

}
