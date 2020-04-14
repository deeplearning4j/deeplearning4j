/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.rl4j.helper;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * INDArray helper methods used by RL4J
 *
 * @author Alexandre Boulanger
 */
public class INDArrayHelper {
    /**
     * MultiLayerNetwork and ComputationGraph expect the first dimension to be the number of examples in the INDArray.
     * In the case of RL4J, it must be 1. This method will return a INDArray with the correct shape.
     *
     * @param source A INDArray
     * @return The source INDArray with the correct shape
     */
    public static INDArray forceCorrectShape(INDArray source) {
        return source.shape()[0] == 1 && source.shape().length > 1
                ? source
                : Nd4j.expandDims(source, 0);
    }
}
