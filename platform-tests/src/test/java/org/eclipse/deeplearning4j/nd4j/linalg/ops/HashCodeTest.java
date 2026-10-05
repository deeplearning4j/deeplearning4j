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
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

/**
 * hashcode is a tree of polynomial hashes over the elements of the array in C order: blocks of 32 consecutive elements
 * hash to one value each (r = 31 * r + v from r = 1), blocks of 32 of those values to one value each, and so on until
 * one is left. Both backends took the chunk starts from the number of chunks, not the chunk size, and returned the
 * hash of the level below the top: an array of 33 to 1024 elements hashed to the hash of its first 32. They hashed the
 * memory of the array, not its elements, so an F-ordered array or a view hashed differently from its C-order copy, and
 * above 1024 elements the two backends combined the upper levels differently.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class HashCodeTest extends BaseNd4jTestWithBackends {

    private static final DataType[] TYPES = {DataType.INT8, DataType.INT32, DataType.INT64, DataType.FLOAT,
            DataType.DOUBLE, DataType.HALF, DataType.BFLOAT16};

    private static long hashcode(INDArray x) {
        return Nd4j.exec(DynamicCustomOp.builder("hashcode").addInputs(x).build())[0].getLong(0);
    }

    /** The values the hash combines: integers as they are, floating point numbers as their bits. */
    private static long[] bytesOf(INDArray array) {
        INDArray c = array.dup('c');
        long[] bytes = new long[(int) c.length()];
        switch (c.dataType()) {
            case FLOAT:
            case HALF:
            case BFLOAT16: {
                float[] values = c.data().asFloat();
                for (int i = 0; i < bytes.length; i++)
                    bytes[i] = Float.floatToRawIntBits(values[i]);
                break;
            }
            case DOUBLE: {
                double[] values = c.data().asDouble();
                for (int i = 0; i < bytes.length; i++)
                    bytes[i] = Double.doubleToRawLongBits(values[i]);
                break;
            }
            default: {
                long[] values = c.data().asLong();
                System.arraycopy(values, 0, bytes, 0, bytes.length);
            }
        }
        return bytes;
    }

    /** One level of the tree: each block of 32 values hashes to one. */
    private static long[] blockHashes(long[] values) {
        long[] hashes = new long[(values.length + 31) / 32];
        for (int b = 0; b < hashes.length; b++) {
            long r = 1;
            for (int e = b * 32; e < Math.min(values.length, b * 32 + 32); e++)
                r = 31 * r + values[e];
            hashes[b] = r;
        }
        return hashes;
    }

    private static long expectedHash(INDArray array) {
        long[] level = blockHashes(bytesOf(array));
        while (level.length > 1)
            level = blockHashes(level);
        return level[0];
    }

    /** A vector of n values, as the type holds them. */
    private static INDArray vector(DataType type, int n, int seed) {
        long[] values = new long[n];
        for (int i = 0; i < n; i++)
            values[i] = ((long) i * 37 + seed * 11 + (i % 7) * 5) % 101 - 50;
        INDArray as64 = Nd4j.createFromArray(values);
        if (type == DataType.FLOAT || type == DataType.DOUBLE || type == DataType.HALF || type == DataType.BFLOAT16)
            return as64.castTo(DataType.DOUBLE).muli(0.37).addi(0.123).castTo(type);
        return as64.castTo(type);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void hashOfEveryLengthIsTheTreeOfItsBlocks(Nd4jBackend backend) {
        int[] lengths = {1, 2, 31, 32, 33, 64, 65, 1000, 1024, 1025, 1056, 5000, 32768, 32769, 100000};
        for (DataType type : TYPES) {
            for (int n : lengths) {
                INDArray x = vector(type, n, n);
                assertEquals(expectedHash(x), hashcode(x), type + " vector of " + n + " elements");
            }
        }
    }

    /** Past a million elements the upper levels need more than one block of threads. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void hashOfPastAMillionElements(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.INT32, DataType.FLOAT}) {
            INDArray x = vector(type, 1_100_003, 5);
            assertEquals(expectedHash(x), hashcode(x), type + " vector of 1100003 elements");
        }
    }

    /** Every element counts: changing or exchanging any one of them changes the hash. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void everyElementChangesTheHash(Nd4jBackend backend) {
        for (int n : new int[]{40, 1000, 5000, 40000}) {
            INDArray x = vector(DataType.INT32, n, 3);
            long hash = hashcode(x);
            for (int index : new int[]{0, 31, 32, 33, n / 2, n - 33, n - 32, n - 1}) {
                INDArray changed = x.dup();
                changed.putScalar(index, changed.getInt(index) + 1);
                assertNotEquals(hash, hashcode(changed), n + " elements, element " + index + " changed");
            }
            // the same values in another order are another array
            INDArray exchanged = x.dup();
            int a = 3;
            int b = n - 4;
            while (x.getInt(b) == x.getInt(a))
                b--;
            exchanged.putScalar(a, x.getInt(b));
            exchanged.putScalar(b, x.getInt(a));
            assertNotEquals(hash, hashcode(exchanged), n + " elements, two exchanged");
        }
    }

    /** The hash is that of the elements in C order: F order, transposes and views hash as their C-order copies. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void hashFollowsTheElementsNotTheMemory(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.INT32, DataType.FLOAT, DataType.DOUBLE}) {
            INDArray matrix = vector(type, 37 * 41, 9).reshape(37, 41);
            INDArray cube = vector(type, 5 * 6 * 7 * 8, 4).reshape(5, 6, 7, 8);
            INDArray parent = vector(type, 80 * 90, 2).reshape(80, 90);
            INDArray[] layouts = {
                    matrix,
                    matrix.dup('f'),
                    matrix.transpose(),
                    cube.dup('f'),
                    cube.permute(3, 0, 2, 1),
                    parent.get(NDArrayIndex.interval(3, 43), NDArrayIndex.interval(5, 50)),
                    parent.get(NDArrayIndex.interval(0, 2, 80), NDArrayIndex.interval(1, 3, 90)),
                    parent.get(NDArrayIndex.point(7), NDArrayIndex.all()),
                    parent.getColumn(11),
            };
            for (int i = 0; i < layouts.length; i++) {
                INDArray layout = layouts[i];
                assertEquals(expectedHash(layout), hashcode(layout),
                        type + " layout " + i + " " + Arrays.toString(layout.shape()) + " order " + layout.ordering());
                assertEquals(hashcode(layout.dup('c')), hashcode(layout),
                        type + " layout " + i + " equals its C-order copy");
            }
        }
    }
}
