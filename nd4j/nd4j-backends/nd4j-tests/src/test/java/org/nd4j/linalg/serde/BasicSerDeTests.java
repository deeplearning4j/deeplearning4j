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

package org.nd4j.linalg.serde;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;


@Slf4j
public class BasicSerDeTests extends BaseNd4jTestWithBackends {

    DataType initialType = Nd4j.dataType();

    @AfterEach
    public void after() {
        Nd4j.setDataType(this.initialType);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBasicDataTypeSwitch1(Nd4jBackend backend) throws Exception {
        DataType initialType = Nd4j.dataType();
        Nd4j.setDataType(DataType.FLOAT);


        INDArray array = Nd4j.create(new float[] {1, 2, 3, 4, 5, 6});

        ByteArrayOutputStream bos = new ByteArrayOutputStream();

        Nd4j.write(bos, array);


        Nd4j.setDataType(DataType.DOUBLE);


        INDArray restored = Nd4j.read(new ByteArrayInputStream(bos.toByteArray()));

        assertEquals(Nd4j.create(new float[] {1, 2, 3, 4, 5, 6}), restored);

        assertEquals(4, restored.data().getElementSize());
        assertEquals(8, restored.shapeInfoDataBuffer().getElementSize());



        Nd4j.setDataType(initialType);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testHalfSerde_1(Nd4jBackend backend) throws Exception {
        val array = Nd4j.create(DataType.HALF, 3, 4);
        array.assign(1.0f);

        val bos = new ByteArrayOutputStream();

        Nd4j.write(bos, array);

        val restored = Nd4j.read(new ByteArrayInputStream(bos.toByteArray()));

        assertEquals(array, restored);
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
