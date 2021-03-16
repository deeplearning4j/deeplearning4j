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

package org.nd4j.linalg.broadcast;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LessThan;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.RealDivOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Slf4j

public class BasicBroadcastTests extends BaseNd4jTestWithBackends {

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void basicBroadcastTest_1(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.FLOAT, 3, 5);
        val y = Nd4j.createFromArray(new float[]{1.f, 1.f, 1.f, 1.f, 1.f});
        val e = Nd4j.create(DataType.FLOAT, 3, 5).assign(1.f);

        // inplace setup
        val op = new AddOp(new INDArray[]{x, y}, new INDArray[]{x});

        Nd4j.exec(op);

        assertEquals(e, x);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void basicBroadcastTest_2(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2);
        val y = Nd4j.createFromArray(new float[]{1.f, 1.f, 1.f, 1.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.FLOAT, 3, 2, 2).assign(1.f);

        //Nd4j.exec(new PrintVariable(x, "X array"));
        //Nd4j.exec(new PrintVariable(y, "Y array"));

        val z = x.add(y);

        assertEquals(e, z);
    }


    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void basicBroadcastTest_3(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(1);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.FLOAT, 3, 2, 2).assign(2.f);

        val z = x.mul(y);

        assertEquals(e, z);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void basicBroadcastTest_4(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.FLOAT, 3, 2, 2).assign(2.f);

        val z = x.div(y);

        assertEquals(e, z);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void basicBroadcastTest_5(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.FLOAT, 3, 2, 2).assign(2.f);

        val z = x.sub(y);

        assertEquals(e, z);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void basicBroadcastTest_6(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.FLOAT, 3, 2, 2).assign(-2.f);

        val z = x.rsub(y);

        assertEquals(e, z);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void basicBroadcastTest_7(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.BOOL, 3, 2, 2).assign(false);

        val z = x.lt(y);

        assertEquals(e, z);
    }

    @Test()
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void basicBroadcastFailureTest_1(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class,() -> {
            val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
            val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
            val z = x.subi(y);
        });
    }

    @Test()
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void basicBroadcastFailureTest_2(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class,() -> {
            val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
            val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
            val z = x.divi(y);
        });

    }

    @Test()
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void basicBroadcastFailureTest_3(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class, () -> {
            val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
            val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
            val z = x.muli(y);
        });

    }

    @Test()
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void basicBroadcastFailureTest_4(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val z = x.addi(y);
    }

    @Test()
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void basicBroadcastFailureTest_5(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class,() -> {
            val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
            val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
            val z = x.rsubi(y);
        });

    }

    @Test()
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void basicBroadcastFailureTest_6(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class,() -> {
            val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
            val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
            val z = x.rdivi(y);
        });

    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void basicBroadcastTest_8(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.BOOL, 3, 2, 2).assign(true);

        val z = x.gt(y);

        assertEquals(e, z);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void basicBroadcastTest_9(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(2.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.BOOL, 3, 2, 2).assign(true);

        val z = x.eq(y);

        assertEquals(e, z);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void basicBroadcastTest_10(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(1.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.BOOL, 3, 2, 2).assign(false);

        val z = x.eq(y);

        assertEquals(e, z);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void emptyBroadcastTest_1(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.FLOAT, 1, 2);
        val y = Nd4j.create(DataType.FLOAT, 0, 2);

        val z = x.add(y);
        assertEquals(y, z);
    }

    @Test()
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void emptyBroadcastTest_2(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.FLOAT, 1, 2);
        val y = Nd4j.create(DataType.FLOAT, 0, 2);

        val z = x.addi(y);
        assertEquals(y, z);

    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void emptyBroadcastTest_3(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.FLOAT, 1, 0, 1);
        val y = Nd4j.create(DataType.FLOAT, 1, 0, 2);

        val op = new RealDivOp(new INDArray[]{x, y}, new INDArray[]{});
        val z = Nd4j.exec(op)[0];

        assertEquals(y, z);
    }


    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testValidInvalidBroadcast(Nd4jBackend backend){
        INDArray x = Nd4j.rand(3,1);
        INDArray y = Nd4j.create(3, 4);

        x.add(y);
        y.addi(x);
        try {
            x.addi(y);
        } catch (Exception e){
            String s = e.getMessage();
            assertTrue(s.contains("broadcast") && s.contains("shape"),s);
        }

        x.sub(y);
        y.subi(x);
        try {
            x.subi(y);
        } catch (Exception e){
            String s = e.getMessage();
            assertTrue(s.contains("broadcast") && s.contains("shape"),s);
        }

        x.mul(y);
        y.muli(x);
        try {
            x.muli(y);
        } catch (Exception e){
            String s = e.getMessage();
            assertTrue(s.contains("broadcast") && s.contains("shape"),s);
        }

        x.div(y);
        y.divi(x);
        try {
            x.divi(y);
        } catch (Exception e){
            String s = e.getMessage();
            assertTrue(s.contains("broadcast") && s.contains("shape"),s);
        }

        x.rsub(y);
        y.rsubi(x);
        try {
            x.rsubi(y);
        } catch (Exception e){
            String s = e.getMessage();
            assertTrue(s.contains("broadcast") && s.contains("shape"),s);
        }

        x.rdiv(y);
        y.rdivi(x);
        try {
            x.rdivi(y);
        } catch (Exception e){
            String s = e.getMessage();
            assertTrue(s.contains("broadcast") && s.contains("shape"),s);
        }
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testLt(Nd4jBackend backend){
        INDArray x = Nd4j.scalar(0);
        INDArray y = Nd4j.createFromArray(2,1,2);

        INDArray result = Nd4j.create(DataType.BOOL, 3);
        INDArray lt = Nd4j.exec(new LessThan(x,y,result))[0];

        INDArray exp = Nd4j.createFromArray(true, true, true);
        assertEquals(exp, lt);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testAdd(Nd4jBackend backend){
        INDArray x = Nd4j.scalar(0);
        INDArray y = Nd4j.createFromArray(2,1,2);

        INDArray result = Nd4j.create(DataType.INT, 3);
        INDArray sum = Nd4j.exec(new AddOp(x,y,result))[0];

        INDArray exp = Nd4j.createFromArray(2, 1, 2);
        assertEquals(exp, sum);
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testBroadcatableBool_1(Nd4jBackend backend) {
        val op = DynamicCustomOp.builder("greater_equal")
                .addInputs(Nd4j.create(DataType.FLOAT, 3), Nd4j.create(DataType.FLOAT, 3))
                .build();

        val l = op.calculateOutputShape();
        assertEquals(1, l.size());
        assertEquals(DataType.BOOL, l.get(0).dataType());
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
