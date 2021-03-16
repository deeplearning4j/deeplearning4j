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

package org.nd4j.linalg.shape.concat.padding;

import lombok.val;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author Adam Gibson
 */

public class PaddingTestsC extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testPrepend(Nd4jBackend backend) {
        INDArray linspace = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray assertion = Nd4j.create(new double[][] {{1, 1, 1, 1, 2}, {1, 1, 1, 3, 4}});

        INDArray prepend = Nd4j.prepend(linspace, 3, 1.0, -1);
        assertEquals(assertion, prepend);


    }


    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testPaddingOneThrougFour(Nd4jBackend backend) {
        int ph = 0;
        int pw = 0;
        int sy = 2;
        int sx = 2;
        INDArray ret = Nd4j.create(new double[] {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4, 4, 4}, new int[] {1, 1, 8, 8});
        INDArray padded = Nd4j.pad(ret, new int[][] {{0, 0}, {0, 0}, {ph, ph + sy - 1}, {pw, pw + sx - 1}});

        INDArray assertion = Nd4j.create(new double[] {1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 3, 3, 3,
                        3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0,
                        3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                new int[] {1, 1, 9, 9});
        assertArrayEquals(assertion.shape(), padded.shape());
        assertEquals(assertion, padded);

    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testAppend2(Nd4jBackend backend) {
        INDArray ret = Nd4j.create(new double[] {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4, 4, 4}, new int[] {1, 1, 8, 8});

        INDArray appendAssertion = Nd4j.create(new double[] {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {1, 1, 9, 8});

        INDArray appended = Nd4j.append(ret, 1, 0, 2);
        assertArrayEquals(appendAssertion.shape(), appended.shape());
        assertEquals(appendAssertion, appended);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testPaddingTensor(Nd4jBackend backend) {
        //,1,1,1,1,2,2,0
        int kh = 1, kw = 1, sy = 1, sx = 1, ph = 2, pw = 2;
        INDArray linspaced = Nd4j.linspace(1, 16, 16, DataType.DOUBLE).reshape(2, 2, 2, 2);
        val n = linspaced.size(0);
        val c = linspaced.size(1);
        val h = linspaced.size(2);
        val w = linspaced.size(3);

        long outWidth = Convolution.outSize(h, kh, sy, ph, 1, true);
        long outHeight = Convolution.outSize(w, kw, sx, pw, 1, true);
        INDArray padded = Nd4j.pad(linspaced, new int[][] {{0, 0}, {0, 0}, {ph, ph + sy - 1}, {pw, pw + sx - 1}});
//        System.out.println(padded);
    }



    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testAppend(Nd4jBackend backend) {
        INDArray linspace = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray otherAppend = Nd4j.append(linspace, 3, 1.0, -1);
        INDArray assertion = Nd4j.create(new double[][] {{1, 2, 1, 1, 1}, {3, 4, 1, 1, 1}});

        assertEquals(assertion, otherAppend);
    }
}
