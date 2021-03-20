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

package org.nd4j.linalg.api.blas;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;

@NativeTag
public class Level2Test extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemv1(Nd4jBackend backend) {
        INDArray array1 = Nd4j.linspace(1, 1000, 1000).reshape(10, 100);
        INDArray array2 = Nd4j.linspace(1, 100, 100).reshape(100, 1);

        INDArray array3 = array1.mmul(array2);

        assertEquals(10, array3.length());
        assertEquals(338350f, array3.getFloat(0), 0.001f);
        assertEquals(843350f, array3.getFloat(1), 0.001f);
        assertEquals(1348350f, array3.getFloat(2), 0.001f);
        assertEquals(1853350f, array3.getFloat(3), 0.001f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemv2(Nd4jBackend backend) {
        INDArray array1 = Nd4j.linspace(1, 1000, 1000).reshape(10, 100);
        INDArray array2 = Nd4j.linspace(1, 100, 100).reshape('f', 100, 1);

        INDArray array3 = array1.mmul(array2);

        assertEquals(10, array3.length());
        assertEquals(338350f, array3.getFloat(0), 0.001f);
        assertEquals(843350f, array3.getFloat(1), 0.001f);
        assertEquals(1348350f, array3.getFloat(2), 0.001f);
        assertEquals(1853350f, array3.getFloat(3), 0.001f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemv3(Nd4jBackend backend) {
        INDArray array1 = Nd4j.linspace(1, 1000, 1000).reshape('f', 10, 100);
        INDArray array2 = Nd4j.linspace(1, 100, 100).reshape('f', 100, 1);

        INDArray array3 = array1.mmul(array2);

        assertEquals(10, array3.length());
        assertEquals(3338050f, array3.getFloat(0), 0.001f);
        assertEquals(3343100f, array3.getFloat(1), 0.001f);
        assertEquals(3348150f, array3.getFloat(2), 0.001f);
        assertEquals(3353200f, array3.getFloat(3), 0.001f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemv4(Nd4jBackend backend) {
        INDArray array1 = Nd4j.linspace(1, 1000, 1000).reshape('f', 10, 100);
        INDArray array2 = Nd4j.linspace(1, 100, 100).reshape(100, 1);

        INDArray array3 = array1.mmul(array2);

        assertEquals(10, array3.length());
        assertEquals(3338050f, array3.getFloat(0), 0.001f);
        assertEquals(3343100f, array3.getFloat(1), 0.001f);
        assertEquals(3348150f, array3.getFloat(2), 0.001f);
        assertEquals(3353200f, array3.getFloat(3), 0.001f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemv5(Nd4jBackend backend) {
        INDArray array1 = Nd4j.linspace(1, 1000, 1000).reshape(10, 100);
        INDArray array2 = Nd4j.linspace(1, 100, 100).reshape(100, 1);

        INDArray array3 = Nd4j.create(10,1);

        array1.mmul(array2, array3);

        assertEquals(10, array3.length());
        assertEquals(338350f, array3.getFloat(0), 0.001f);
        assertEquals(843350f, array3.getFloat(1), 0.001f);
        assertEquals(1348350f, array3.getFloat(2), 0.001f);
        assertEquals(1853350f, array3.getFloat(3), 0.001f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemv6(Nd4jBackend backend) {
        INDArray array1 = Nd4j.linspace(1, 1000, 1000).reshape('f', 10, 100);
        INDArray array2 = Nd4j.linspace(1, 100, 100).reshape(100, 1);

        INDArray array3 = Nd4j.create(10,1);

        array1.mmul(array2, array3);

        assertEquals(10, array3.length());
        assertEquals(3338050f, array3.getFloat(0), 0.001f);
        assertEquals(3343100f, array3.getFloat(1), 0.001f);
        assertEquals(3348150f, array3.getFloat(2), 0.001f);
        assertEquals(3353200f, array3.getFloat(3), 0.001f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemv7(Nd4jBackend backend) {
        INDArray array1 = Nd4j.linspace(1, 1000, 1000).reshape('f', 10, 100);
        INDArray array2 = Nd4j.linspace(1, 100, 100).reshape(100, 1);

        INDArray array3 = Nd4j.create(10,1);

        array1.mmul(array2, array3);

        assertEquals(10, array3.length());
        assertEquals(3338050f, array3.getFloat(0), 0.001f);
        assertEquals(3343100f, array3.getFloat(1), 0.001f);
        assertEquals(3348150f, array3.getFloat(2), 0.001f);
        assertEquals(3353200f, array3.getFloat(3), 0.001f);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
