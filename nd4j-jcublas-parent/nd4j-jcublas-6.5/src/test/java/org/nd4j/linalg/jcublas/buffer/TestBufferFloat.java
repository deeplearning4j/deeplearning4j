/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.jcublas.buffer;


import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.test.FloatDataBufferTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.*;
import static org.junit.Assert.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Created by agibsonccc on 2/14/15.
 */
public class TestBufferFloat extends FloatDataBufferTest {

    private static Logger log = LoggerFactory.getLogger(TestBufferFloat.class);

    private void write(INDArray arr, String s) throws Exception {
        FileOutputStream fos = new FileOutputStream(s);
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(arr);
        oos.close();
    }

    private INDArray read(String s) throws Exception {
        FileInputStream fis = new FileInputStream(s);
        ObjectInputStream ois = new ObjectInputStream(fis);
        INDArray arr = (INDArray) ois.readObject();
        ois.close();
        return arr;
    }

    @Test
    public void testAsFloat() {
        float[] answer = new float[]{1,1,1,1};
        DataBuffer buff = Nd4j.createBuffer(new float[]{1,1,1,1});
        assertTrue(Arrays.equals(answer, buff.asFloat()));
        buff.destroy();
    }



    @Test
    public void testSerializationFloat() {
        Path p, p2;
        try {
            p = Files.createTempFile("ndarray", "ser");
            p2 = Files.createTempFile("ndarray.empty", "ser");
        }
        catch (IOException ex) {
            log.error("Couldn't create temporary file. Skip testSerializationFloat");
            return;
        }

        float[] arrData = new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float[] arrData2 = new float[] {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
        float eps = 1e-6f;

        try {
            Nd4j.dtype = DataBuffer.FLOAT;

            INDArray arr = Nd4j.create(arrData, new int[] {2, 3});
            INDArray arrEmpty = Nd4j.create(new float[0], new int[] {0});

            arr.muli(2.0f);

            Assert.assertArrayEquals(arr.data().asFloat(), arrData2, eps);
            Assert.assertEquals(arrEmpty.data(), null);
            Assert.assertArrayEquals(arrEmpty.shape(), new int[] {0});

            write(arr, p.toString());
            write(arrEmpty, p2.toString());
        }
        catch (Exception ex)
        {
            Assert.fail("Exception thrown during test: " + ex.toString());
        }

        try
        {
            INDArray arr = read(p.toString());
            INDArray arrEmpty = read(p2.toString());

            Assert.assertArrayEquals(arr.shape(), new int[]{2, 3});
            Assert.assertArrayEquals(arr.data().asFloat(), arrData2, eps);

            arr.divi(2.0f);
            Assert.assertArrayEquals(arr.data().asFloat(), arrData, eps);

            Assert.assertEquals(arrEmpty.data(), null);
            Assert.assertArrayEquals(arrEmpty.shape(), new int[] {0});

            // Clean up the file
            Files.delete(p);
            Files.delete(p2);
        }
        catch (Exception ex)
        {
            Assert.fail("Exception thrown during test: " + ex.toString());
        }
    }
}
