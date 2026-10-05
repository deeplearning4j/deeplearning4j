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
package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import org.bytedeco.javacpp.IntPointer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.DataTypeEx;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Constant buffers and buffer type conversions keep doubles and 64-bit integers exact. The native element conversion
 * behind them went through float for every pair of types, so a DOUBLE constant 0.3 became 0.30000001192092896 (a
 * Vulkan dropout's p, every legacy op's extra arguments), an INT64 constant above 2^24 lost its low bits, and a
 * DOUBLE-to-DOUBLE conversion rounded to float.
 * <p>
 * The element codes of NDArrayFactory.convertDataEx are DataTypeEx ordinals on every backend. The native conversion
 * read them as DataType values, and DataTypeEx.DOUBLE's ordinal is DataType INT8's: a DOUBLE-to-DOUBLE conversion
 * copied one byte per element (0.3 read back as 1.09e-312), and on CUDA a host loop ran over the device buffers.
 */
@NativeTag
@Tag(TagNames.COMPRESSION)
public class TypeConversionPrecisionTest extends BaseNd4jTestWithBackends {

    private static final double[] DOUBLES = {0.3, 1e300, 1.0 + Math.ulp(1.0), -4.9e-324, 123456789.123456789};
    private static final long[] LONGS = {(1L << 53) + 1, Long.MAX_VALUE, Long.MIN_VALUE, -(1L << 40) - 3, 16777217};

    /** Values every element type the conversion codes name holds exactly. */
    private static final double[] SMALL = {0, 1, 2, 7, 64, 100, 127};
    private static final DataTypeEx[] ELEMENT_CODES = {DataTypeEx.INT8, DataTypeEx.UINT8, DataTypeEx.FLOAT16,
            DataTypeEx.INT16, DataTypeEx.UINT16, DataTypeEx.FLOAT, DataTypeEx.DOUBLE};

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void constantBuffersKeepTheirValues(Nd4jBackend backend) {
        DataBuffer doubles = Nd4j.getConstantHandler().getConstantBuffer(DOUBLES, DataType.DOUBLE);
        for (int i = 0; i < DOUBLES.length; i++)
            assertEquals(Double.doubleToLongBits(DOUBLES[i]), Double.doubleToLongBits(doubles.getDouble(i)),
                    "DOUBLE constant " + DOUBLES[i] + " read back as " + doubles.getDouble(i));

        DataBuffer longs = Nd4j.getConstantHandler().getConstantBuffer(LONGS, DataType.INT64);
        for (int i = 0; i < LONGS.length; i++)
            assertEquals(LONGS[i], longs.getLong(i), "INT64 constant");

        int[] ints = {16777217, Integer.MAX_VALUE, Integer.MIN_VALUE};
        DataBuffer intBuffer = Nd4j.getConstantHandler().getConstantBuffer(ints, DataType.INT32);
        for (int i = 0; i < ints.length; i++)
            assertEquals(ints[i], intBuffer.getInt(i), "INT32 constant");

        // integers become doubles exactly up to 2^53
        long[] exactInDouble = {1L << 53, 16777217, -(1L << 40) - 3};
        DataBuffer widened = Nd4j.getConstantHandler().getConstantBuffer(exactInDouble, DataType.DOUBLE);
        for (int i = 0; i < exactInDouble.length; i++)
            assertEquals((double) exactInDouble[i], widened.getDouble(i), 0.0, "INT64 constant as DOUBLE");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void doubleConversionsKeepTheirValues(Nd4jBackend backend) {
        // the buffer conversion (the INDArray form compresses and decompresses)
        DataBuffer source = Nd4j.createBuffer(DOUBLES);
        DataBuffer target = Nd4j.createBuffer(DataType.DOUBLE, DOUBLES.length, false);
        Nd4j.getNDArrayFactory().convertDataEx(DataTypeEx.DOUBLE, source, DataTypeEx.DOUBLE, target);
        double[] values = target.asDouble();
        for (int i = 0; i < DOUBLES.length; i++)
            assertEquals(Double.doubleToLongBits(DOUBLES[i]), Double.doubleToLongBits(values[i]),
                    "DOUBLE to DOUBLE " + DOUBLES[i] + " became " + values[i]);
    }

    /** Every pair of element codes converts each value to the same value in the target's type. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void everyElementCodePairConverts(Nd4jBackend backend) {
        for (DataTypeEx from : ELEMENT_CODES) {
            for (DataTypeEx to : ELEMENT_CODES) {
                DataBuffer source = Nd4j.createBuffer(elementType(from), SMALL.length, false);
                for (int i = 0; i < SMALL.length; i++)
                    source.put(i, SMALL[i]);
                DataBuffer target = Nd4j.createBuffer(elementType(to), SMALL.length, false);
                Nd4j.getNDArrayFactory().convertDataEx(from, source, to, target);
                assertEquals(elementType(to), target.dataType(), from + " to " + to + " target type");
                for (int i = 0; i < SMALL.length; i++)
                    assertEquals(SMALL[i], target.getDouble(i), 0.0, from + " to " + to + " element " + i);
            }
        }
    }

    /**
     * THRESHOLD encodes the elements at or beyond the threshold as signed one-based indices, taking the threshold out of
     * them, and decoding adds plus or minus the threshold back at those indices: a host encoding, which CUDA (device
     * buffers) and Vulkan reject.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void thresholdEncodingRoundTrip(Nd4jBackend backend) {
        float threshold = 0.5f;
        float[] values = {0.1f, 0.7f, -0.6f, 0.2f, 1.6f, -0.4f, -2.0f, 0.0f};
        int[] expectedEncoding = {-7, -3, 2, 5};
        DataBuffer dense = Nd4j.createBuffer(values);
        try (IntPointer encoding = new IntPointer(expectedEncoding.length + 4)) {
            encoding.put(0, expectedEncoding.length);
            encoding.put(1, values.length);
            encoding.put(2, Float.floatToIntBits(threshold));
            encoding.put(3, 0);

            OpExecutioner.ExecutionerType type = Nd4j.getExecutioner().type();
            if (type == OpExecutioner.ExecutionerType.CUDA || type == OpExecutioner.ExecutionerType.VULKAN) {
                assertThrows(RuntimeException.class, () -> Nd4j.getNDArrayFactory().convertDataEx(DataTypeEx.FLOAT,
                        dense.addressPointer(), DataTypeEx.THRESHOLD, encoding, values.length));
                return;
            }

            Nd4j.getNDArrayFactory().convertDataEx(DataTypeEx.FLOAT, dense.addressPointer(), DataTypeEx.THRESHOLD,
                    encoding, values.length);
            assertEquals(values.length, encoding.get(1), "encoded length");
            int[] entries = new int[expectedEncoding.length];
            for (int i = 0; i < entries.length; i++)
                entries[i] = encoding.get(4 + i);
            Arrays.sort(entries);
            assertArrayEquals(expectedEncoding, entries, "encoded indices");
            for (int e = 0; e < values.length; e++) {
                float v = values[e];
                float residual = v >= threshold ? v - threshold : v <= -threshold ? v + threshold : v;
                assertEquals(residual, dense.getFloat(e), 0.0f, "residual at " + e);
            }

            DataBuffer decoded = Nd4j.createBuffer(DataType.FLOAT, values.length, true);
            Nd4j.getNDArrayFactory().convertDataEx(DataTypeEx.THRESHOLD, encoding, DataTypeEx.FLOAT, decoded);
            float[] expected = new float[values.length];
            for (int entry : expectedEncoding)
                expected[Math.abs(entry) - 1] = entry > 0 ? threshold : -threshold;
            assertArrayEquals(expected, decoded.asFloat(), 0.0f, "decoded updates");
        }
    }

    private static DataType elementType(DataTypeEx code) {
        switch (code) {
            case INT8:
                return DataType.INT8;
            case UINT8:
                return DataType.UINT8;
            case FLOAT16:
                return DataType.HALF;
            case INT16:
                return DataType.INT16;
            case UINT16:
                return DataType.UINT16;
            case FLOAT:
                return DataType.FLOAT;
            case DOUBLE:
                return DataType.DOUBLE;
            default:
                throw new IllegalArgumentException("not an element code: " + code);
        }
    }
}
