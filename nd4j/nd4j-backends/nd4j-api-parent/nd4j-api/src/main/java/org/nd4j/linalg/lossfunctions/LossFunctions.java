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

package org.nd4j.linalg.lossfunctions;

import org.nd4j.linalg.lossfunctions.impl.*;


/**
 * Central class for loss functions
 * @author Adam Gibson
 */
public class LossFunctions {

    /**
     * MSE: Mean Squared Error: Linear Regression - {@link LossMSE}<br>
     * l1: L1 loss (absolute value) - {@link LossL1}<br>
     * XENT: Cross Entropy: Binary Classification - {@link LossBinaryXENT}<br>
     * MCXENT: Multiclass Cross Entropy - {@link LossMCXENT}<br>
     * SPARSE_MCXENT: Sparse multi-class cross entropy - {@link LossSparseMCXENT}<br>
     * SQUARED_LOSS: Alias for mean squared error - {@link LossMSE}<br>
     * NEGATIVELOGLIKELIHOOD: Negative Log Likelihood - {@link LossNegativeLogLikelihood}<br>
     * COSINE_PROXIMITY: Cosine proximity loss - {@link LossCosineProximity}<br>
     * HINGE: Hinge loss - {@link LossHinge}<br>
     * SQUARED_HINGE: Squared hinge loss - {@link LossSquaredHinge}<br>
     * KL_DIVERGENCE: Kullback-Leibler divergence loss - {@link LossKLD}<br>
     * MEAN_ABSOLUTE_ERROR: mean absolute error loss - {@link LossMAE}<br>
     * L2: L2 loss (sum of squared errors) - {@link LossL2}<br>
     * MEAN_ABSOLUTE_PERCENTAGE_ERROR: MAPE loss - {@link LossMAPE}<br>
     * MEAN_SQUARED_LOGARITHMIC_ERROR: MSLE loss - {@link LossMSLE}<br>
     * POISSON: Poisson loss - {@link LossPoisson}<br>
     * WASSERSTEIN: Wasserstein loss - {@link LossWasserstein}
     */
    public enum LossFunction {
        MSE, L1, XENT, MCXENT, SPARSE_MCXENT, SQUARED_LOSS, RECONSTRUCTION_CROSSENTROPY, NEGATIVELOGLIKELIHOOD, COSINE_PROXIMITY, HINGE,
        SQUARED_HINGE, KL_DIVERGENCE, MEAN_ABSOLUTE_ERROR, L2, MEAN_ABSOLUTE_PERCENTAGE_ERROR, MEAN_SQUARED_LOGARITHMIC_ERROR, POISSON, WASSERSTEIN;

        public ILossFunction getILossFunction() {
            switch (this) {
                case MSE:
                case SQUARED_LOSS:
                    return new LossMSE();
                case L1:
                    return new LossL1();
                case XENT:
                    return new LossBinaryXENT();
                case MCXENT:
                    return new LossMCXENT();
                case SPARSE_MCXENT:
                    return new LossSparseMCXENT();
                case KL_DIVERGENCE:
                case RECONSTRUCTION_CROSSENTROPY:
                    return new LossKLD();
                case NEGATIVELOGLIKELIHOOD:
                    return new LossNegativeLogLikelihood();
                case COSINE_PROXIMITY:
                    return new LossCosineProximity();
                case HINGE:
                    return new LossHinge();
                case SQUARED_HINGE:
                    return new LossSquaredHinge();
                case MEAN_ABSOLUTE_ERROR:
                    return new LossMAE();
                case L2:
                    return new LossL2();
                case MEAN_ABSOLUTE_PERCENTAGE_ERROR:
                    return new LossMAPE();
                case MEAN_SQUARED_LOGARITHMIC_ERROR:
                    return new LossMSLE();
                case POISSON:
                    return new LossPoisson();
                case WASSERSTEIN:
                    return new LossWasserstein();
                default:
                    //Custom, RMSE_XENT
                    throw new UnsupportedOperationException("Unknown or not supported loss function: " + this);
            }
        }
    }


}
