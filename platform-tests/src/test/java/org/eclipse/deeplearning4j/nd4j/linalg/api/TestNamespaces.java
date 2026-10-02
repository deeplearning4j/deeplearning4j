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

package org.eclipse.deeplearning4j.nd4j.linalg.api;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;
@NativeTag
public class TestNamespaces extends BaseNd4jTestWithBackends {


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBitwiseSimple(Nd4jBackend backend){
        for (DataType type : new DataType[]{DataType.INT, DataType.LONG}) {
            INDArray x = Nd4j.createFromArray(6, 3, 12, 5, 255).reshape(1, 5).castTo(type);
            INDArray y = Nd4j.createFromArray(3, 5, 10, 4, 15).reshape(1, 5).castTo(type);
            assertEquals(Nd4j.createFromArray(2, 1, 8, 4, 15).reshape(1, 5).castTo(type), Nd4j.bitwise.and(x, y));
            assertEquals(Nd4j.createFromArray(7, 7, 14, 5, 255).reshape(1, 5).castTo(type), Nd4j.bitwise.or(x, y));
            assertEquals(Nd4j.createFromArray(5, 6, 6, 1, 240).reshape(1, 5).castTo(type), Nd4j.bitwise.xor(x, y));

            INDArray shifts = Nd4j.createFromArray(1, 2, 0, 3, 4).reshape(1, 5).castTo(type);
            assertEquals(Nd4j.createFromArray(12, 12, 12, 40, 4080).reshape(1, 5).castTo(type),
                    Nd4j.bitwise.leftShift(x, shifts));
            assertEquals(Nd4j.createFromArray(3, 0, 12, 0, 15).reshape(1, 5).castTo(type),
                    Nd4j.bitwise.rightShift(x, shifts));

            // Against a scalar, as a loop condition combines a [1] comparison with its extra condition.
            INDArray one = Nd4j.createFromArray(1).reshape(1).castTo(type);
            INDArray scalarOne = Nd4j.scalar(1).castTo(type);
            assertEquals(one, Nd4j.bitwise.and(one, scalarOne));
            assertEquals(one, Nd4j.bitwise.or(one, scalarOne));
            assertEquals(Nd4j.createFromArray(0).reshape(1).castTo(type), Nd4j.bitwise.xor(one, scalarOne));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMathSimple(Nd4jBackend backend) {
        INDArray x = Nd4j.rand(DataType.FLOAT, 1, 5).muli(2).subi(1);
        INDArray abs = Nd4j.math.abs(x);
//        System.out.println(x);
//        System.out.println(abs);


        INDArray c1 = Nd4j.createFromArray(0, 2, 1);
        INDArray c2 = Nd4j.createFromArray(1, 2, 1);

        INDArray cm = Nd4j.math.confusionMatrix(c1, c2, 3);
//        System.out.println(cm);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRandomSimple(Nd4jBackend backend){
        INDArray normal = Nd4j.random.normal(0, 1, DataType.FLOAT, 10);
//        System.out.println(normal);
        INDArray uniform = Nd4j.random.uniform(0, 1, DataType.FLOAT, 10);
//        System.out.println(uniform);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNeuralNetworkSimple(Nd4jBackend backend){
        INDArray out = Nd4j.nn.elu(Nd4j.random.normal(0, 1, DataType.FLOAT, 10));
        INDArray out2 = Nd4j.nn.softmax(Nd4j.random.normal(0, 1, DataType.FLOAT, 4, 5), 1);
    }

    @Override
    public char ordering() {
        return 'c';
    }

}
