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

package org.nd4j.weightinit;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Defines weight initialization for neural networks.
 *
 * Use {@link BaseWeightInitScheme}
 * to create a new {@link WeightInitScheme}
 * This is needed to  handle things like the parameters view.
 *
 * @author Adam Gibson
 */
public interface WeightInitScheme {

    /**
     * Create the array
     * @param shape the shape of the array
     * @param paramsView the parameters view
     * @return the created array
     */
    INDArray create(long[] shape,INDArray paramsView);



    /**
     * Create the array
     * @param shape the shape of the array
     * @return the created array
     */
    INDArray create(long... shape);


    /**
     * The order of the weight init
     * @return
     */
    char order();

    /**
     * The type of the weight init
     * @return
     */
    WeightInit type();

}
