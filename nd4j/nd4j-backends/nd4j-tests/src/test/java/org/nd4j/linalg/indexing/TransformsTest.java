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

package org.nd4j.linalg.indexing;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Slf4j
@NativeTag
public class TransformsTest extends BaseNd4jTestWithBackends {



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEq1(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new double[] {0, 1, 2, 1});
        INDArray exp = Nd4j.create(new boolean[] {false, false, true, false});

        INDArray z = x.eq(2);

        assertEquals(exp, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNEq1(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new double[] {0, 1, 2, 1});
        INDArray exp = Nd4j.create(new boolean[] {true, false, true, false});

        INDArray z = x.neq(1);

        assertEquals(exp, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLT1(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new double[] {0, 1, 2, 1});
        INDArray exp = Nd4j.create(new boolean[] {true, true, false, true});

        INDArray z = x.lt(2);

        assertEquals(exp, z);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGT1(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new double[] {0, 1, 2, 4});
        INDArray exp = Nd4j.create(new boolean[] {false, false, true, true});

        INDArray z = x.gt(1);

        assertEquals(exp, z);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarMinMax1(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new double[] {1, 3, 5, 7});
        INDArray xCopy = x.dup();
        INDArray exp1 = Nd4j.create(new double[] {1, 3, 5, 7});
        INDArray exp2 = Nd4j.create(new double[] {1e-5, 1e-5, 1e-5, 1e-5});

        INDArray z1 = Transforms.max(x, Nd4j.EPS_THRESHOLD, true);
        INDArray z2 = Transforms.min(x, Nd4j.EPS_THRESHOLD, true);

        assertEquals(exp1, z1);
        assertEquals(exp2, z2);
        // Assert that x was not modified
        assertEquals(x, xCopy);

        INDArray exp3 = Nd4j.create(new double[] {10, 10, 10, 10});
        Transforms.max(x, 10, false);
        assertEquals(exp3, x);

        Transforms.min(x, Nd4j.EPS_THRESHOLD, false);
        assertEquals(exp2, x);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArrayMinMax(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new double[] {1, 3, 5, 7});
        INDArray y = Nd4j.create(new double[] {2, 2, 6, 6});
        INDArray xCopy = x.dup();
        INDArray yCopy = y.dup();
        INDArray expMax = Nd4j.create(new double[] {2, 3, 6, 7});
        INDArray expMin = Nd4j.create(new double[] {1, 2, 5, 6});

        INDArray z1 = Transforms.max(x, y, true);
        INDArray z2 = Transforms.min(x, y, true);

        assertEquals(expMax, z1);
        assertEquals(expMin, z2);
        // Assert that x was not modified
        assertEquals(xCopy, x);

        Transforms.max(x, y, false);
        // Assert that x was modified
        assertEquals(expMax, x);
        // Assert that y was not modified
        assertEquals(yCopy, y);

        // Reset the modified x
        x = xCopy.dup();

        Transforms.min(x, y, false);
        // Assert that X was modified
        assertEquals(expMin, x);
        // Assert that y was not modified
        assertEquals(yCopy, y);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAnd1(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new double[] {0, 0, 1, 0, 0});
        INDArray y = Nd4j.create(new double[] {0, 0, 1, 1, 0});
        INDArray e = Nd4j.create(new boolean[] {false, false, true, false, false});

        INDArray z = Transforms.and(x, y);

        assertEquals(e, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOr1(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new double[] {0, 0, 1, 0, 0});
        INDArray y = Nd4j.create(new double[] {0, 0, 1, 1, 0});
        val e = Nd4j.create(new boolean[] {false, false, true, true, false});

        INDArray z = Transforms.or(x, y);

        assertEquals(e, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testXor1(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new double[] {0, 0, 1, 0, 0});
        INDArray y = Nd4j.create(new double[] {0, 0, 1, 1, 0});
        INDArray exp = Nd4j.create(new boolean[] {false, false, false, true, false});

        INDArray z = Transforms.xor(x, y);

        assertEquals(exp, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNot1(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new double[] {0, 0, 1, 0, 0});
        INDArray exp = Nd4j.create(new boolean[] {false, false, true, false, false});

        INDArray z = Transforms.not(x);

        assertEquals(exp, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSlice_1(Nd4jBackend backend) {
        val arr = Nd4j.linspace(1,4, 4, DataType.FLOAT).reshape(2, 2, 1);
        val exp0 = Nd4j.create(new float[]{1, 2}, new int[] {2, 1});
        val exp1 = Nd4j.create(new float[]{3, 4}, new int[] {2, 1});

        val slice0 = arr.slice(0).dup('c');
        assertEquals(exp0, slice0);
        assertEquals(exp0, arr.slice(0));

        val slice1 = arr.slice(1).dup('c');
        assertEquals(exp1, slice1);
        assertEquals(exp1, arr.slice(1));

        val tf = arr.slice(1);
        val slice1_1 = tf.slice(0);
        assertTrue(slice1_1.isScalar());
        assertEquals(3.0, slice1_1.getDouble(0), 1e-5);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
