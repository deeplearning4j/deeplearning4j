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

package org.eclipse.deeplearning4j.nd4j.linalg.serde;

import lombok.extern.slf4j.Slf4j;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.*;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;


@Slf4j
@Tag(TagNames.JACKSON_SERDE)
@NativeTag
@Tag(TagNames.LARGE_RESOURCES)
@Tag(TagNames.LONG_TEST)
public class LargeSerDeTests extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLargeArraySerDe_1(Nd4jBackend backend) throws Exception {
        var arrayA = Nd4j.rand(new long[] {1, 135079944});
        //var arrayA = Nd4j.rand(new long[] {1, 13507});

        var tmpFile = File.createTempFile("sdsds", "sdsd");
        tmpFile.deleteOnExit();

        try (var fos = new FileOutputStream(tmpFile); var bos = new BufferedOutputStream(fos); var dos = new DataOutputStream(bos)) {
            Nd4j.write(arrayA, dos);
        }


        try (var fis = new FileInputStream(tmpFile); var bis = new BufferedInputStream(fis); var dis = new DataInputStream(bis)) {
            var arrayB = Nd4j.read(dis);

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

        var tmpFile = File.createTempFile("sdsds", "sdsd");
        tmpFile.deleteOnExit();

        log.info("Starting serialization...");
        var sS = System.currentTimeMillis();
        try (var fos = new FileOutputStream(tmpFile); var bos = new BufferedOutputStream(fos); var dos = new DataOutputStream(bos)) {
            Nd4j.write(arrayA, dos);
            arrayA = null;
            System.gc();
        }
        System.gc();

        var sE = System.currentTimeMillis();

        log.info("Starting deserialization...");
        var dS = System.currentTimeMillis();
        try (var fis = new FileInputStream(tmpFile); var bis = new BufferedInputStream(fis); var dis = new DataInputStream(bis)) {
            arrayA = Nd4j.read(dis);
        }
        var dE = System.currentTimeMillis();

        log.info("Timings: {Ser : {} ms; De: {} ms;}", sE - sS, dE - dS);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
