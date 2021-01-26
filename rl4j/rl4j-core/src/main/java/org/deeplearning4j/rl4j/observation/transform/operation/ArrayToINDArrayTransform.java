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
package org.deeplearning4j.rl4j.observation.transform.operation;

import org.datavec.api.transform.Operation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A simple transform that converts a double[] into a INDArray
 */
public class ArrayToINDArrayTransform implements Operation<double[], INDArray> {
    private final long[] shape;

    /**
     * @param shape Reshapes the INDArrays with this shape
     */
    public ArrayToINDArrayTransform(long... shape) {
        this.shape = shape;
    }

    /**
     * Will construct 1-D INDArrays
     */
    public ArrayToINDArrayTransform() {
        this.shape = null;
    }

    @Override
    public INDArray transform(double[] data) {
        INDArray result = Nd4j.create(data);
        if(shape != null) {
            result = result.reshape(shape);
        }
        return result;
    }
}
