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

package org.deeplearning4j.nn.adapters;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.nn.api.OutputAdapter;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * This OutputAdapter implementation takes single 2D nn output in, and returns JVM double[][] array
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class Regression2dAdapter implements OutputAdapter<double[][]> {
    @Override
    public double[][] apply(INDArray... outputs) {
        Preconditions.checkArgument(outputs.length == 1, "Argmax adapter can have only 1 output");
        val array = outputs[0];
        Preconditions.checkArgument(array.rank() < 3, "Argmax adapter requires 2D or 1D output");

        if (array.rank() == 2 && !array.isVector()) {
            return array.toDoubleMatrix();
        } else {
            val result = new double[1][(int) array.length()];

            for (int e = 0; e< array.length(); e++)
                result[0][e] = array.getDouble(e);

            return result;
        }
    }
}
