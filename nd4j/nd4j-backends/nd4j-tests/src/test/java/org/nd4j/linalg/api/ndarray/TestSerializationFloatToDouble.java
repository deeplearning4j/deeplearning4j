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

package org.nd4j.linalg.api.ndarray;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.*;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;


public class TestSerializationFloatToDouble extends BaseNd4jTestWithBackends {

    DataType initialType = Nd4j.dataType();


    @AfterEach
    public void after() {
        Nd4j.setDataType(this.initialType);
    }

      @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSerializationFullArrayNd4jWriteRead(Nd4jBackend backend) throws Exception {
        int length = 100;

        //WRITE OUT A FLOAT ARRAY
        //Hack before setting datatype - fix already in r119_various branch
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataType.FLOAT);
        INDArray arr = Nd4j.linspace(1, length, length).reshape('c', 10, 10);
        arr.subi(50.0123456); //assures positive and negative numbers with decimal points

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (DataOutputStream dos = new DataOutputStream(baos)) {
            Nd4j.write(arr, dos);
        }
        byte[] bytes = baos.toByteArray();

        //SET DATA TYPE TO DOUBLE and initialize another array with the same contents
        //Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        System.out.println("The data opType is " + Nd4j.dataType());
        INDArray arr1 = Nd4j.linspace(1, length, length, DataType.FLOAT).reshape('c', 10, 10);
        arr1.subi(50.0123456);

        INDArray arr2;
        try (DataInputStream dis = new DataInputStream(new ByteArrayInputStream(bytes))) {
            arr2 = Nd4j.read(dis);
        }

        //System.out.println(new NDArrayStrings(6).format(arr2.sub(arr1).mul(100).div(arr1)));
        assertTrue(Transforms.abs(arr1.sub(arr2).div(arr1)).maxNumber().doubleValue() < 0.01);
    }

      @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSerializationFullArrayJava() throws Exception {
        int length = 100;
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataType.FLOAT);
        INDArray arr = Nd4j.linspace(1, length, length).reshape('c', 10, 10);
        arr.subi(50.0123456); //assures positive and negative numbers with decimal points

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(arr);
        }
        byte[] bytes = baos.toByteArray();

        //SET DATA TYPE TO DOUBLE and initialize another array with the same contents
        //Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        System.out.println("The data opType is " + Nd4j.dataType());
        INDArray arr1 = Nd4j.linspace(1, length, length).reshape('c', 10, 10);
        arr1.subi(50.0123456);

        INDArray arr2;
        try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bytes))) {
            arr2 = (INDArray) ois.readObject();
        }

        assertEquals(arr.dataType(), arr2.dataType());
        arr1 = arr1.castTo(arr2.dataType());
        assertTrue(Transforms.abs(arr1.sub(arr2).div(arr1)).maxNumber().doubleValue() < 0.01);
    }

      @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSerializationOnViewsNd4jWriteRead() throws Exception {
        int length = 100;
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataType.FLOAT);
        INDArray arr = Nd4j.linspace(1, length, length).reshape('c', 10, 10);
        INDArray sub = arr.get(NDArrayIndex.interval(5, 10), NDArrayIndex.interval(5, 10));

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (DataOutputStream dos = new DataOutputStream(baos)) {
            Nd4j.write(sub, dos);
        }
        byte[] bytes = baos.toByteArray();

        //SET DATA TYPE TO DOUBLE and initialize another array with the same contents
        //Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        System.out.println("The data opType is " + Nd4j.dataType());
        INDArray arr1 = Nd4j.linspace(1, length, length, DataType.FLOAT).reshape('c', 10, 10);
        INDArray sub1 = arr1.get(NDArrayIndex.interval(5, 10), NDArrayIndex.interval(5, 10));

        INDArray arr2;
        try (DataInputStream dis = new DataInputStream(new ByteArrayInputStream(bytes))) {
            arr2 = Nd4j.read(dis);
        }

        //assertEquals(sub,arr2);\
        assertTrue(Transforms.abs(sub1.sub(arr2).div(sub1)).maxNumber().doubleValue() < 0.01);
    }

      @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSerializationOnViewsJava() throws Exception {
        int length = 100;
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataType.FLOAT);
        INDArray arr = Nd4j.linspace(1, length, length).reshape('c', 10, 10);

        INDArray sub = arr.get(NDArrayIndex.interval(5, 10), NDArrayIndex.interval(5, 10));

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(sub);
        }
        byte[] bytes = baos.toByteArray();
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        System.out.println("The data opType is " + Nd4j.dataType());
        INDArray arr1 = Nd4j.linspace(1, length, length, DataType.FLOAT).reshape('c', 10, 10);
        INDArray sub1 = arr1.get(NDArrayIndex.interval(5, 10), NDArrayIndex.interval(5, 10));

        INDArray arr2;
        try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bytes))) {
            arr2 = (INDArray) ois.readObject();
        }

        //assertEquals(sub,arr2);
        assertTrue(Transforms.abs(sub1.sub(arr2).div(sub1)).maxNumber().doubleValue() < 0.01);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}

