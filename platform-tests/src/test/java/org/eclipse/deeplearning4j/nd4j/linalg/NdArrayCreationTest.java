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

package org.eclipse.deeplearning4j.nd4j.linalg;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.nio.ByteBuffer;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests focused on NDArray creation: constructors, factory methods, buffer creation,
 * data type handling, and shape validation.
 */
@Slf4j
@NativeTag
@Tag(TagNames.FILE_IO)
public class NdArrayCreationTest extends BaseNd4jTestWithBackends {

    @TempDir
    Path testDir;

    @BeforeEach
    public void before() throws Exception {
        Nd4j.getRandom().setSeed(123);
        Nd4j.getExecutioner().enableDebugMode(false);
        Nd4j.getExecutioner().enableVerboseMode(false);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmptyStringScalar(Nd4jBackend backend) {
        INDArray arr = Nd4j.empty(DataType.UTF8);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArangeNegative(Nd4jBackend backend) {
        INDArray arr = Nd4j.arange(-2, 2).castTo(DataType.DOUBLE);
        INDArray assertion = Nd4j.create(new double[]{-2, -1, 0, 1});
        assertEquals(assertion, arr);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateDetached_1(Nd4jBackend backend) {
        val shape = new int[]{10};
        val dataTypes = new DataType[]{DataType.DOUBLE, DataType.BOOL, DataType.BYTE, DataType.UBYTE, DataType.SHORT, DataType.UINT16, DataType.INT, DataType.UINT32, DataType.LONG, DataType.UINT64, DataType.FLOAT, DataType.BFLOAT16, DataType.HALF};

        for (DataType dt : dataTypes) {
            val dataBuffer = Nd4j.createBufferDetached(shape, dt);
            assertEquals(dt, dataBuffer.dataType());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateDetached_2(Nd4jBackend backend) {
        val shape = new long[]{10};
        val dataTypes = new DataType[]{DataType.DOUBLE, DataType.BOOL, DataType.BYTE, DataType.UBYTE, DataType.SHORT, DataType.UINT16, DataType.INT, DataType.UINT32, DataType.LONG, DataType.UINT64, DataType.FLOAT, DataType.BFLOAT16, DataType.HALF};

        for (DataType dt : dataTypes) {
            val dataBuffer = Nd4j.createBufferDetached(shape, dt);
            assertEquals(dt, dataBuffer.dataType());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateUnitialized(Nd4jBackend backend) {

        INDArray arrC = Nd4j.createUninitialized(new long[]{10, 10}, 'c');
        INDArray arrF = Nd4j.createUninitialized(new long[]{10, 10}, 'f');

        assertEquals('c', arrC.ordering());
        assertArrayEquals(new long[]{10, 10}, arrC.shape());
        assertEquals('f', arrF.ordering());
        assertArrayEquals(new long[]{10, 10}, arrF.shape());

        //Can't really test that it's *actually* uninitialized...
        arrC.assign(0);
        arrF.assign(0);

        assertEquals(Nd4j.create(new long[]{10, 10}), arrC);
        assertEquals(Nd4j.create(new long[]{10, 10}), arrF);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testValueArrayOf_1(Nd4jBackend backend) {
        val vector = Nd4j.valueArrayOf(new long[]{5}, 2f, DataType.FLOAT);
        val exp = Nd4j.createFromArray(new float[]{2, 2, 2, 2, 2});

        assertArrayEquals(exp.shape(), vector.shape());
        assertEquals(exp, vector);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testValueArrayOf_2(Nd4jBackend backend) {
        val scalar = Nd4j.valueArrayOf(new long[]{}, 2f);
        val exp = Nd4j.scalar(2f);

        assertArrayEquals(exp.shape(), scalar.shape());
        assertEquals(exp, scalar);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArrayCreation(Nd4jBackend backend) {
        val vector = Nd4j.create(new float[]{1, 2, 3}, new long[]{3}, 'c');
        val exp = Nd4j.createFromArray(new float[]{1, 2, 3});

        assertArrayEquals(exp.shape(), vector.shape());
        assertEquals(exp, vector);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testHalfStuff(Nd4jBackend backend) {
        assumeTrue(Nd4j.getExecutioner().getClass().getSimpleName().toLowerCase().contains("cuda"),
            "Half-precision ops require a CUDA backend");

        val dtype = Nd4j.dataType();
        Nd4j.setDataType(DataType.HALF);

        val arr = Nd4j.ones(3, 3);
        arr.addi(2.0f);

        val exp = Nd4j.create(3, 3).assign(3.0f);

        assertEquals(exp, arr);

        Nd4j.setDataType(dtype);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRndBloat16(Nd4jBackend backend) {
        INDArray x = Nd4j.rand(DataType.BFLOAT16, 'c', new long[]{5});
        assertTrue(x.sumNumber().floatValue() > 0);

        x = Nd4j.randn(DataType.BFLOAT16, 10);
        assertTrue(x.sumNumber().floatValue() != 0.0);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateF() {
        char origOrder = Nd4j.order();
        try {
            Nd4j.factory().setOrder('f');

            INDArray arr = Nd4j.createFromArray(new double[][]{{1, 2, 3}, {4, 5, 6}});
            INDArray arr2 = Nd4j.createFromArray(new float[][]{{1, 2, 3}, {4, 5, 6}});
            INDArray arr3 = Nd4j.createFromArray(new int[][]{{1, 2, 3}, {4, 5, 6}});
            INDArray arr4 = Nd4j.createFromArray(new long[][]{{1, 2, 3}, {4, 5, 6}});
            INDArray arr5 = Nd4j.createFromArray(new short[][]{{1, 2, 3}, {4, 5, 6}});
            INDArray arr6 = Nd4j.createFromArray(new byte[][]{{1, 2, 3}, {4, 5, 6}});

            INDArray exp = Nd4j.create(2, 3);
            exp.putScalar(0, 0, 1.0);
            exp.putScalar(0, 1, 2.0);
            exp.putScalar(0, 2, 3.0);
            exp.putScalar(1, 0, 4.0);
            exp.putScalar(1, 1, 5.0);
            exp.putScalar(1, 2, 6.0);

            assertEquals(exp, arr);
            assertEquals(exp.castTo(DataType.FLOAT), arr2);
            assertEquals(exp.castTo(DataType.INT), arr3);
            assertEquals(exp.castTo(DataType.LONG), arr4);
            assertEquals(exp.castTo(DataType.SHORT), arr5);
            assertEquals(exp.castTo(DataType.BYTE), arr6);
        } finally {
            Nd4j.factory().setOrder(origOrder);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOnes(Nd4jBackend backend) {
        INDArray arr = Nd4j.ones();
        INDArray arr2 = Nd4j.ones(DataType.LONG);
        assertEquals(0, arr.rank());
        assertEquals(1, arr.length());
        assertEquals(0, arr2.rank());
        assertEquals(1, arr2.length());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testZeros(Nd4jBackend backend) {
        INDArray arr = Nd4j.zeros();
        INDArray arr2 = Nd4j.zeros(DataType.LONG);
        assertEquals(0, arr.rank());
        assertEquals(1, arr.length());
        assertEquals(0, arr2.rank());
        assertEquals(1, arr2.length());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateDtypes(Nd4jBackend backend) {
        int[] sliceShape = new int[]{9};
        float[] arrays = new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        double[] arrays_double = new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

        INDArray x = Nd4j.create(sliceShape, arrays, arrays);
        assertEquals(DataType.FLOAT, x.dataType());

        INDArray xd = Nd4j.create(sliceShape, arrays_double, arrays_double);
        assertEquals(DataType.DOUBLE, xd.dataType());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateShapeValidation() {
        try {
            Nd4j.create(new double[]{1, 2, 3}, new int[]{1, 1});
            fail();
        } catch (Exception t) {
            assertTrue(t.getMessage().contains("length"));
        }

        try {
            Nd4j.create(new float[]{1, 2, 3}, new int[]{1, 1});
            fail();
        } catch (Exception t) {
            assertTrue(t.getMessage().contains("length"));
        }

        try {
            Nd4j.create(new byte[]{1, 2, 3}, new long[]{1, 1}, DataType.BYTE);
            fail();
        } catch (Exception t) {
            assertTrue(t.getMessage().contains("length"));
        }

        try {
            Nd4j.create(new double[]{1, 2, 3}, new int[]{1, 1}, 'c');
            fail();
        } catch (Exception t) {
            assertTrue(t.getMessage().contains("length"));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LARGE_RESOURCES)
    @Tag(TagNames.LONG_TEST)
    public void testCreateBufferFromByteBuffer(Nd4jBackend backend) {

        for (DataType dt : DataType.values()) {
            // Skip types with no fixed width (string types, compressed, unknown) and
            // FLOAT8/FLOAT8_E5M2 which require HAS_FLOAT8 native build support
            if (dt == DataType.COMPRESSED || dt == DataType.UTF8 || dt == DataType.UNKNOWN
                    || dt == DataType.UTF16 || dt == DataType.UTF32
                    || dt == DataType.FLOAT8 || dt == DataType.FLOAT8_E5M2)
                continue;

            int lengthBytes = 256;
            int lengthElements = lengthBytes / dt.width();
            ByteBuffer bb = ByteBuffer.allocateDirect(lengthBytes);

            DataBuffer db = Nd4j.createBuffer(bb, dt, lengthElements);
            INDArray arr = Nd4j.create(db, new long[]{lengthElements});

            arr.toStringFull();
            arr.toString();

            for (DataType dt2 : DataType.values()) {
                // Skip types not supported: string types, compressed, unknown, and
                // FLOAT8/FLOAT8_E5M2 which require HAS_FLOAT8 native build support
                if (dt2 == DataType.COMPRESSED || dt2 == DataType.UTF8 || dt2 == DataType.UNKNOWN
                        || dt2 == DataType.UTF16 || dt2 == DataType.UTF32
                        || dt2 == DataType.FLOAT8 || dt2 == DataType.FLOAT8_E5M2)
                    continue;
                INDArray a2 = arr.castTo(dt2);
                a2.toStringFull();
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateBufferFromByteBufferViews() {

        for (DataType dt : DataType.values()) {
            // Skip types with no fixed width (string types, compressed, unknown) and
            // FLOAT8/FLOAT8_E5M2 which require HAS_FLOAT8 native build support
            if (dt == DataType.COMPRESSED || dt == DataType.UTF8 || dt == DataType.UNKNOWN
                    || dt == DataType.UTF16 || dt == DataType.UTF32
                    || dt == DataType.FLOAT8 || dt == DataType.FLOAT8_E5M2)
                continue;

            int lengthBytes = 256;
            int lengthElements = lengthBytes / dt.width();
            ByteBuffer bb = ByteBuffer.allocateDirect(lengthBytes);

            DataBuffer db = Nd4j.createBuffer(bb, dt, lengthElements);
            INDArray arr = Nd4j.create(db, new long[]{lengthElements / 2, 2});

            arr.toStringFull();

            INDArray view = arr.get(org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.point(0));
            INDArray view2 = arr.get(org.nd4j.linalg.indexing.NDArrayIndex.point(1), org.nd4j.linalg.indexing.NDArrayIndex.all());

            view.toStringFull();
            view2.toStringFull();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShape0Casts() {
        for (DataType dt : DataType.values()) {
            if (!dt.isNumerical())
                continue;
            // FLOAT8/FLOAT8_E5M2 require HAS_FLOAT8 native build support; skip if not present
            if (dt == DataType.FLOAT8 || dt == DataType.FLOAT8_E5M2)
                continue;

            INDArray a1 = Nd4j.create(dt, 1, 0, 2);

            for (DataType dt2 : DataType.values()) {
                if (!dt2.isNumerical())
                    continue;
                // FLOAT8/FLOAT8_E5M2 require HAS_FLOAT8 native build support; skip if not present
                if (dt2 == DataType.FLOAT8 || dt2 == DataType.FLOAT8_E5M2)
                    continue;
                INDArray a2 = a1.castTo(dt2);

                assertArrayEquals(a1.shape(), a2.shape());
                assertEquals(dt2, a2.dataType());
            }
        }
    }
}
