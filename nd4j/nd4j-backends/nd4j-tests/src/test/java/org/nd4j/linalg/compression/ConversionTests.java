/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.compression;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

@Slf4j
@RunWith(Parameterized.class)
public class ConversionTests extends BaseNd4jTest {

    public ConversionTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    DataBuffer.Type initialType;

    @After
    public void after() {
        Nd4j.setDataType(this.initialType);
    }


    @Test
    public void testDoubleToFloats1() {
        val dtype = Nd4j.dataType();

        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        val arrayX = Nd4j.create(10).assign(1.0);


        Nd4j.setDataType(DataBuffer.Type.FLOAT);
        val arrayY = Nd4j.create(10).assign(1.0);


        val converted = arrayX.convertToFloats();
        val exp = Nd4j.create(10).assign(2.0);
        converted.addi(arrayY);

        assertEquals(exp, converted);



        Nd4j.setDataType(dtype);
    }


    @Test
    public void testFloatsToDoubles1() {
        val dtype = Nd4j.dataType();

        Nd4j.setDataType(DataBuffer.Type.FLOAT);
        val arrayX = Nd4j.create(10).assign(1.0);


        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        val arrayY = Nd4j.create(10).assign(1.0);


        val converted = arrayX.convertToDoubles();
        val exp = Nd4j.create(10).assign(2.0);
        converted.addi(arrayY);

        assertEquals(exp, converted);



        Nd4j.setDataType(dtype);
    }

    @Test
    public void testFloatsToHalfs1() {
        if (!Nd4j.getExecutioner().getClass().getSimpleName().toLowerCase().contains("cuda"))
            return;

        val dtype = Nd4j.dataType();

        Nd4j.setDataType(DataBuffer.Type.FLOAT);
        val arrayX = Nd4j.create(10).assign(1.0);


        Nd4j.setDataType(DataBuffer.Type.HALF);
        val arrayY = Nd4j.create(10).assign(1.0);


        val converted = arrayX.convertToHalfs();
        val exp = Nd4j.create(10).assign(2.0);
        converted.addi(arrayY);

        assertEquals(exp, converted);



        Nd4j.setDataType(dtype);
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
