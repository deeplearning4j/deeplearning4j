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

package org.nd4j.linalg;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.*;

import static org.junit.jupiter.api.Assertions.assertEquals;


@Slf4j
public class DataTypeTest extends BaseNd4jTestWithBackends {

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testDataTypes(Nd4jBackend backend) throws Exception {
        for (val type : DataType.values()) {
            if (DataType.UTF8.equals(type) || DataType.UNKNOWN.equals(type) || DataType.COMPRESSED.equals(type))
                continue;

            val in1 = Nd4j.ones(type, 10, 10);

            val baos = new ByteArrayOutputStream();
            val oos = new ObjectOutputStream(baos);
            try {
                oos.writeObject(in1);
            } catch (Exception e) {
                throw new RuntimeException("Failed for data type [" + type + "]", e);
            }

            val bios = new ByteArrayInputStream(baos.toByteArray());
            val ois = new ObjectInputStream(bios);
            try {
                val in2 = (INDArray) ois.readObject();
                assertEquals( in1, in2,"Failed for data type [" + type + "]");
            } catch (Exception e) {
                throw new RuntimeException("Failed for data type [" + type + "]", e);
            }
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
