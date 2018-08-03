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
import org.apache.commons.lang3.SerializationUtils;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.ByteArrayOutputStream;

@Slf4j
@Ignore
@RunWith(Parameterized.class)
public class CompressionPerformanceTests extends BaseNd4jTest {

    public CompressionPerformanceTests(Nd4jBackend backend) {
            super(backend);
    }


    @Test
    public void groundTruthTests_Threshold_1() {
        Nd4j.getRandom().setSeed(119);
        val params = Nd4j.rand(new long[]{1, 50000000}, -1.0, 1.0, Nd4j.getRandom());
        val original = params.dup(params.ordering());

        int iterations = 1000;

        long timeE = 0;
        long timeS = 0;
        long timeD = 0;
        int s = 0;
        for (int e = 0; e < iterations; e++) {
            val timeES = System.currentTimeMillis();
            val encoded = Nd4j.getExecutioner().thresholdEncode(params, 0.99);
            val timeEE = System.currentTimeMillis();


            params.assign(original);
            timeE += (timeEE - timeES);


            val bs = new ByteArrayOutputStream();
            val timeSS = System.currentTimeMillis();
            SerializationUtils.serialize(encoded, bs);
            val timeSE = System.currentTimeMillis();

            timeS += (timeSE - timeSS);

            val ba = bs.toByteArray();
            val timeDS = System.currentTimeMillis();
            val deserialized = SerializationUtils.deserialize(ba);
            val timeDE = System.currentTimeMillis();
            timeD += (timeDE - timeDS);

            s = bs.size();
        }


        log.info("Encoding time: {} ms; Serialization time: {} ms; Deserialized time: {} ms; Serialized bytes: {}", timeE / iterations, timeS / iterations, timeD / iterations, s);
    }

    @Test
    public void groundTruthTests_Bitmap_1() {
        Nd4j.getRandom().setSeed(119);
        val params = Nd4j.rand(new long[]{1, 25000000}, -1.0, 1.0, Nd4j.getRandom());
        val original = params.dup(params.ordering());

        int iterations = 1000;

        DataBuffer buffer = Nd4j.getDataBufferFactory().createInt(params.length() / 16 + 5);

        INDArray ret = Nd4j.createArrayFromShapeBuffer(buffer, params.shapeInfoDataBuffer());

        long time = 0;
        for (int e = 0; e < iterations; e++) {
            val timeES = System.currentTimeMillis();
            Nd4j.getExecutioner().bitmapEncode(params, ret,0.99);
            val timeEE = System.currentTimeMillis();


            params.assign(original);
            Nd4j.getMemoryManager().memset(ret);
            time += (timeEE - timeES);
        }


        log.info("Encoding time: {} ms;", time / iterations);
    }

        @Override
    public char ordering() {
        return 'c';
    }
}
