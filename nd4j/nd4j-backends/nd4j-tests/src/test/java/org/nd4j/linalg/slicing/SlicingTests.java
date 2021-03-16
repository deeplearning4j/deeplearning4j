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

package org.nd4j.linalg.slicing;

import lombok.val;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author Adam Gibson
 */

public class SlicingTests extends BaseNd4jTestWithBackends {


    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSlices() {
        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24, DataType.DOUBLE).data(), new int[] {4, 3, 2});
        for (int i = 0; i < arr.slices(); i++) {
            INDArray slice = arr.slice(i).slice(1);
            val slices = slice.slices();
            assertEquals(2, slices);
        }

    }



    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSlice() {
        INDArray arr = Nd4j.linspace(1, 24, 24, DataType.DOUBLE).reshape(4, 3, 2);
        INDArray assertion = Nd4j.create(new double[][] {{1, 13}, {5, 17}, {9, 21}});

        INDArray firstSlice = arr.slice(0);
        INDArray slice1Assertion = Nd4j.create(new double[][] {{2, 14}, {6, 18}, {10, 22},

        });

        INDArray secondSlice = arr.slice(1);
        assertEquals(assertion, firstSlice);
        assertEquals(slice1Assertion, secondSlice);

    }


    @Override
    public char ordering() {
        return 'f';
    }
}
