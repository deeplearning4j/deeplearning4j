/*
 *  ******************************************************************************
 *  *
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

package org.nd4j.linalg.learning.regularization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * Regularization API for magnitude-based regularization techniques such as:<br>
 * {@link L1Regularization}<br>
 * {@link L2Regularization}<br>
 * {@link WeightDecay}<br>
 * <br>
 * Implementations should have the following features:<br>
 * 1. Have a loss function (score) component that is based on the input (usually parameter) array<br>
 * 2. Modify the gradients (or updates) array based on the current input array (parameters)<br>
 * 3. Optionally, use the current learning rate when modifying gradients<br>
 * <br>
 * Note that generally this type of regularization is applied to parameters, but in principle this type of regularization
 * can be applied to activations also.
 * <br>
 * This Regularization interface cannot be used for all types of regularization, however; for example, the API
 * is not appropriate for implementing DropOut/DropConnect regularization.
 *
 * @author Alex Black
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface Regularization extends Serializable {

    /**
     * ApplyStep determines how the regularization interacts with the optimization process - i.e., when it is applied
     * relative to updaters like Adam, Nesterov momentum, SGD, etc.
     * <br>
     * <br>
     * BEFORE_UPDATER: w -= updater(gradient + regularization(p,gradView,lr)) <br>
     * POST_UPDATER: w -= (updater(gradient) + regularization(p,gradView,lr)) <br>
     *
     */
    enum ApplyStep {
        BEFORE_UPDATER,
        POST_UPDATER
    }

    /**
     * @return The step that the regularization should be applied, as defined by {@link ApplyStep}
     */
    ApplyStep applyStep();

    /**
     * Apply the regularization by modifying the gradient array in-place
     *
     * @param param     Input array (usually parameters)
     * @param gradView  Gradient view array (should be modified/updated). Same shape and type as the input array.
     * @param lr        Current learning rate
     * @param iteration Current network training iteration
     * @param epoch     Current network training epoch
     */
    void apply(INDArray param, INDArray gradView, double lr, int iteration, int epoch);

    /**
     * Calculate the loss function score component for the regularization.<br>
     * For example, in L2 regularization, this would return {@code L = 0.5 * sum_i param[i]^2}<br>
     * For regularization types that don't have a score component, this method can return 0. However, note that this may
     * make the regularization type not gradient checkable.
     *
     * @param param     Input array (usually parameters)
     * @param iteration Current network training iteration
     * @param epoch     Current network training epoch
     * @return          Loss function score component based on the input/parameters array
     */
    double score(INDArray param, int iteration, int epoch);

    /**
     * @return An independent copy of the regularization instance
     */
    Regularization clone();

}
