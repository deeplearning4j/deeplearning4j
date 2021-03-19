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
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.*;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;


@Slf4j
@Disabled("AB 2019/05/23 - JVM crash on linux-x86_64-cpu-avx512 - issue #7657")
public class LargeSerDeTests extends BaseNd4jTestWithBackends {

      @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLargeArraySerDe_1(Nd4jBackend backend) throws Exception {
        val arrayA = Nd4j.rand(new long[] {1, 135079944});
        //val arrayA = Nd4j.rand(new long[] {1, 13507});

        val tmpFile = File.createTempFile("sdsds", "sdsd");
        tmpFile.deleteOnExit();

        try (val fos = new FileOutputStream(tmpFile); val bos = new BufferedOutputStream(fos); val dos = new DataOutputStream(bos)) {
            Nd4j.write(arrayA, dos);
        }


        try (val fis = new FileInputStream(tmpFile); val bis = new BufferedInputStream(fis); val dis = new DataInputStream(bis)) {
            val arrayB = Nd4j.read(dis);

            assertArrayEquals(arrayA.shape(), arrayB.shape());
            assertEquals(arrayA.length(), arrayB.length());
            assertEquals(arrayA, arrayB);
        }
    }


    @Test
    @Disabled // this should be commented out, since it requires approx 10GB ram to run
    public void testLargeArraySerDe_2(Nd4jBackend backend) throws Exception {
        INDArray arrayA = Nd4j.createUninitialized(100000, 12500);
        log.info("Shape: {}; Length: {}", arrayA.shape(), arrayA.length());

        val tmpFile = File.createTempFile("sdsds", "sdsd");
        tmpFile.deleteOnExit();

        log.info("Starting serialization...");
        val sS = System.currentTimeMillis();
        try (val fos = new FileOutputStream(tmpFile); val bos = new BufferedOutputStream(fos); val dos = new DataOutputStream(bos)) {
            Nd4j.write(arrayA, dos);
            arrayA = null;
            System.gc();
        }
        System.gc();

        val sE = System.currentTimeMillis();

        log.info("Starting deserialization...");
        val dS = System.currentTimeMillis();
        try (val fis = new FileInputStream(tmpFile); val bis = new BufferedInputStream(fis); val dis = new DataInputStream(bis)) {
            arrayA = Nd4j.read(dis);
        }
        val dE = System.currentTimeMillis();

        log.info("Timings: {Ser : {} ms; De: {} ms;}", sE - sS, dE - dS);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
