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

package org.nd4j.linalg.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author Hamdi Douss
 */

public class NDArrayUtilTest extends BaseNd4jTestWithBackends {


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatrixConversion(Nd4jBackend backend) {
        int[][] nums = {{1, 2}, {3, 4}, {5, 6}};
        INDArray result = NDArrayUtil.toNDArray(nums);
        assertArrayEquals(new long[]{2,3}, result.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorConversion(Nd4jBackend backend) {
        int[] nums = {1, 2, 3, 4};
        INDArray result = NDArrayUtil.toNDArray(nums);
        assertArrayEquals(new long[]{1, 4}, result.shape());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFlattenArray1(Nd4jBackend backend) {
        float[][][] arrX = new float[2][2][2];

        float[] arrZ = ArrayUtil.flatten(arrX);

        assertEquals(8, arrZ.length);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFlattenArray2(Nd4jBackend backend) {
        float[][][] arrX = new float[5][4][3];

        float[] arrZ = ArrayUtil.flatten(arrX);

        assertEquals(60, arrZ.length);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFlattenArray3(Nd4jBackend backend) {
        float[][][] arrX = new float[5][2][3];

        float[] arrZ = ArrayUtil.flatten(arrX);

        assertEquals(30, arrZ.length);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFlattenArray4(Nd4jBackend backend) {
        float[][][][] arrX = new float[5][2][3][3];

        float[] arrZ = ArrayUtil.flatten(arrX);

        assertEquals(90, arrZ.length);
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
