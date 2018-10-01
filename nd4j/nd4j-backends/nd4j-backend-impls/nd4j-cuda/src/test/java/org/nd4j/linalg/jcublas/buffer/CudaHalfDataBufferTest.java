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

package org.nd4j.linalg.jcublas.buffer;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class CudaHalfDataBufferTest {
    private static Logger logger = LoggerFactory.getLogger(CudaHalfDataBufferTest.class);

    @Before
    public void setUp() {
        CudaEnvironment.getInstance().getConfiguration().enableDebug(true);
    }

    @Test
    public void testConversion1() throws Exception {
        DataBuffer bufferOriginal = new CudaFloatDataBuffer(new float[]{1f, 2f, 3f, 4f, 5f});

        DataBuffer bufferHalfs = Nd4j.getNDArrayFactory().convertDataEx(DataTypeEx.FLOAT, bufferOriginal, DataTypeEx.FLOAT16);

        DataBuffer bufferRestored = Nd4j.getNDArrayFactory().convertDataEx(DataTypeEx.FLOAT16, bufferHalfs, DataTypeEx.FLOAT);


        logger.info("Buffer original: {}", Arrays.toString(bufferOriginal.asFloat()));
        logger.info("Buffer restored: {}", Arrays.toString(bufferRestored.asFloat()));

        assertArrayEquals(bufferOriginal.asFloat(), bufferRestored.asFloat(), 0.01f);
    }

    @Test
    public void testSerialization1() throws Exception {
        DataBuffer bufferOriginal = new CudaFloatDataBuffer(new float[]{1f, 2f, 3f, 4f, 5f});

        DataBuffer bufferHalfs = Nd4j.getNDArrayFactory().convertDataEx(DataTypeEx.FLOAT, bufferOriginal, DataTypeEx.FLOAT16);

        File tempFile = File.createTempFile("alpha", "11");
        tempFile.deleteOnExit();

        // now we serialize halfs, and we expect it to become floats on other side
        try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(tempFile.getAbsolutePath())))){
            bufferHalfs.write(dos);
        }

        // loading data back from file
        DataInputStream dis = new DataInputStream(new FileInputStream(tempFile.getAbsoluteFile()));

        DataBuffer bufferRestored = Nd4j.createBuffer(bufferOriginal.length());
        bufferRestored.read(dis);

        assertArrayEquals(bufferOriginal.asFloat(), bufferRestored.asFloat(), 0.01f);
    }

    @Test
    public void testSerialization2() throws Exception {
        DataBuffer bufferOriginal = new CudaFloatDataBuffer(new float[]{1f, 2f, 3f, 4f, 5f});

        DataBuffer bufferHalfs = Nd4j.getNDArrayFactory().convertDataEx(DataTypeEx.FLOAT, bufferOriginal, DataTypeEx.FLOAT16);

        DataTypeUtil.setDTypeForContext(DataType.HALF);

        File tempFile = File.createTempFile("alpha", "11");
        tempFile.deleteOnExit();

        // now we serialize halfs, and we expect it to become floats on other side
        try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(tempFile.getAbsolutePath())))){
            bufferHalfs.write(dos);
        }

        // loading data back from file
        DataInputStream dis = new DataInputStream(new FileInputStream(tempFile.getAbsoluteFile()));

        DataBuffer bufferRestored = Nd4j.createBuffer(bufferOriginal.length());
        bufferRestored.read(dis);

        assertEquals(bufferRestored.dataType(), DataType.HALF);

        DataTypeUtil.setDTypeForContext(DataType.FLOAT);

        DataBuffer bufferConverted = Nd4j.getNDArrayFactory().convertDataEx(DataTypeEx.FLOAT16, bufferRestored, DataTypeEx.FLOAT);

        assertArrayEquals(bufferOriginal.asFloat(), bufferConverted.asFloat(), 0.01f);
    }

    @Test
    public void testSingleConversions() throws Exception {
        CudaFloatDataBuffer bufferOriginal = new CudaFloatDataBuffer(new float[]{1f, 2f, 3f, 4f, 5f});

        short f = bufferOriginal.fromFloat(1.0f);
        float h = bufferOriginal.toFloat((int) f);

        logger.info("Short F: {}, Float F: {}", f, h);

        assertEquals(1.0f, h, 0.001f);
    }
}
