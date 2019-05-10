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

import lombok.val;
import org.deeplearning4j.nn.api.OutputAdapter;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This OutputAdapter implementation is suited for silent conversion of 2D SoftMax output
 *
 * @author raver119@gmail.com
 */
public class ArgmaxAdapter implements OutputAdapter<int[]> {

    /**
     * This method does conversion from INDArrays to int[], where each element will represents position of the highest element in output INDArray
     * I.e. Array of {0.25, 0.1, 0.5, 0.15} will return int array with length of 1, and value {2}
     *
     * @param outputs
     * @return
     */
    @Override
    public int[] apply(INDArray... outputs) {
        Preconditions.checkArgument(outputs.length == 1, "Argmax adapter can have only 1 output");
        val array = outputs[0];
        Preconditions.checkArgument(array.rank() < 3, "Argmax adapter requires 2D or 1D output");
        val result = array.rank() == 2 ? new int[(int) array.size(0)] : new int[1];

        if (array.rank() == 2) {
            val t = Nd4j.argMax(array, 1);
            for (int e = 0; e < t.length(); e++)
                result[e] = (int) t.getDouble(e);
        } else
            result[0] = (int) Nd4j.argMax(array, Integer.MAX_VALUE).getDouble(0);

        return result;
    }
}
