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

package org.nd4j.linalg.shape.ones;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author Adam Gibson
 */

public class LeadingAndTrailingOnesC extends BaseNd4jTestWithBackends {


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateLeadingAndTrailingOnes(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(1, 10, 1, 1);
        arr.assign(1);
//        System.out.println(arr);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatrix(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray slice1 = arr.slice(1);
//        System.out.println(arr.slice(1));
        INDArray oneInMiddle = Nd4j.linspace(1, 4, 4).reshape(2, 1, 2);
        INDArray otherSlice = oneInMiddle.slice(1);
        assertEquals(2, otherSlice.offset());
//        System.out.println(otherSlice);
        INDArray twoOnesInMiddle = Nd4j.linspace(1, 4, 4).reshape(2, 1, 1, 2);
        INDArray sub = twoOnesInMiddle.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.all());
        assertEquals(2, sub.offset());

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultipleOnesInMiddle(Nd4jBackend backend) {
        INDArray tensor = Nd4j.linspace(1, 144, 144).reshape(2, 2, 1, 1, 6, 6);
        INDArray tensorSlice1 = tensor.slice(1);
        INDArray tensorSlice1Slice1 = tensorSlice1.slice(1);
//        System.out.println(tensor);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
