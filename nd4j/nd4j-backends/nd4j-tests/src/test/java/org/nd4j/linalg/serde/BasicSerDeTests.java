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

package org.nd4j.linalg.serde;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.util.SerializationUtils;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

/**
 * Created by raver119 on 21.12.16.
 */
@RunWith(Parameterized.class)
@Slf4j
public class BasicSerDeTests extends BaseNd4jTest {
    public BasicSerDeTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    DataBuffer.Type initialType;

    @After
    public void after() {
        Nd4j.setDataType(this.initialType);
    }


    @Test
    public void testBasicDataTypeSwitch1() throws Exception {
        DataBuffer.Type initialType = Nd4j.dataType();
        Nd4j.setDataType(DataBuffer.Type.FLOAT);


        INDArray array = Nd4j.create(new float[] {1, 2, 3, 4, 5, 6});

        ByteArrayOutputStream bos = new ByteArrayOutputStream();

        Nd4j.write(bos, array);


        Nd4j.setDataType(DataBuffer.Type.DOUBLE);


        INDArray restored = Nd4j.read(new ByteArrayInputStream(bos.toByteArray()));

        assertEquals(Nd4j.create(new float[] {1, 2, 3, 4, 5, 6}), restored);

        assertEquals(8, restored.data().getElementSize());
        assertEquals(8, restored.shapeInfoDataBuffer().getElementSize());



        Nd4j.setDataType(initialType);
    }

    @Test
    public void testSerDe_Threshold_1() throws Exception {
        val original = Nd4j.create(1000000);
        val target = Nd4j.create(1000000);

        for (int e = 0; e < 16; e++)
            original.putScalar(e+ 100, 1e-3);

        val exp = original.dup();

        Nd4j.getExecutioner().commit();

        val encoded = Nd4j.getExecutioner().thresholdEncode(original, 1e-3);

        val bos = new ByteArrayOutputStream();
        SerializationUtils.serialize(encoded, bos);
        log.info("Serialized size: {} bytes", bos.size());


        INDArray deserialized = SerializationUtils.deserialize(bos.toByteArray());


        Nd4j.getExecutioner().thresholdDecode(deserialized, target);

        Nd4j.getExecutioner().commit();

        assertArrayEquals(exp.data().asDouble(), target.data().asDouble(), 1e-5);
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
