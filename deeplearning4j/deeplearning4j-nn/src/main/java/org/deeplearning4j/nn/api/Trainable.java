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

package org.deeplearning4j.nn.api;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Trainable: an interface common to Layers and GraphVertices that have trainable parameters
 *
 * @author Alex Black
 */
public interface Trainable {

    /**
     * @return Training configuration
     */
    TrainingConfig getConfig();

    /**
     * @return Number of parameters
     */
    long numParams();

    /**
     * @return 1d parameter vector
     */
    INDArray params();

    /**
     * @param backpropOnly If true: return only parameters that are not exclusively used for layerwise pretraining
     * @return Parameter table
     */
    Map<String,INDArray> paramTable(boolean backpropOnly);

    /**
     * DL4J layers typically produce the sum of the gradients during the backward pass for each layer, and if required
     * (if minibatch=true) then divide by the minibatch size.<br>
     * However, there are some exceptions, such as the batch norm mean/variance estimate parameters: these "gradients"
     * are actually not gradients, but are updates to be applied directly to the parameter vector. Put another way,
     * most gradients should be divided by the minibatch to get the average; some "gradients" are actually final updates
     * already, and should not be divided by the minibatch size.
     *
     * @param paramName Name of the parameter
     * @return True if gradients should be divided by minibatch (most params); false otherwise (edge cases like batch norm mean/variance estimates)
     */
    boolean updaterDivideByMinibatch(String paramName);

    /**
     * @return 1D gradients view array
     */
    INDArray getGradientsViewArray();

}
