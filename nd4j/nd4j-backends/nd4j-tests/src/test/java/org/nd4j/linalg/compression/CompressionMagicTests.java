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

package org.nd4j.linalg.compression;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.COMPRESSION)
public class CompressionMagicTests extends BaseNd4jTestWithBackends {

    @BeforeEach
    public void setUp() {

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMagicDecompression1(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 100, 2500, DataType.FLOAT);

        INDArray compressed = Nd4j.getCompressor().compress(array, "GZIP");

        assertTrue(compressed.isCompressed());
        compressed.muli(1.0);

        assertFalse(compressed.isCompressed());
        assertEquals(array, compressed);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMagicDecompression4(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 100, 2500, DataType.FLOAT);

        INDArray compressed = Nd4j.getCompressor().compress(array, "GZIP");

        for (int cnt = 0; cnt < array.length(); cnt++) {
            float a = array.getFloat(cnt);
            float c = compressed.getFloat(cnt);
            assertEquals(a, c, 0.01f);
        }

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDupSkipDecompression1(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 100, 2500, DataType.FLOAT);

        INDArray compressed = Nd4j.getCompressor().compress(array, "GZIP");

        INDArray newArray = compressed.dup();
        assertTrue(newArray.isCompressed());

        Nd4j.getCompressor().decompressi(compressed);
        Nd4j.getCompressor().decompressi(newArray);

        assertEquals(array, compressed);
        assertEquals(array, newArray);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDupSkipDecompression2(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 100, 2500, DataType.FLOAT);

        INDArray compressed = Nd4j.getCompressor().compress(array, "GZIP");

        INDArray newArray = compressed.dup('c');
        assertTrue(newArray.isCompressed());

        Nd4j.getCompressor().decompressi(compressed);
        Nd4j.getCompressor().decompressi(newArray);

        assertEquals(array, compressed);
        assertEquals(array, newArray);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDupSkipDecompression3(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 100, 2500, DataType.FLOAT);

        INDArray compressed = Nd4j.getCompressor().compress(array, "GZIP");

        INDArray newArray = compressed.dup('f');
        assertFalse(newArray.isCompressed());

        Nd4j.getCompressor().decompressi(compressed);
        //        Nd4j.getCompressor().decompressi(newArray);

        assertEquals(array, compressed);
        assertEquals(array, newArray);
        assertEquals('f', newArray.ordering());
        assertEquals('c', compressed.ordering());
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
