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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * random_shuffle permutes its input along dimension 0, as TensorFlow does: a vector's elements, a matrix's rows. The
 * CUDA vector path read every element at x[0] (a literal 0 stride), launched Fisher-Yates swapped (1024 blocks of
 * 2^power threads) and merged with 0 blocks; the CPU vector path left the generator where it was, so successive
 * shuffles repeated one permutation, and an in-place row shuffle drew other numbers than an out-of-place one.
 */
@NativeTag
@Tag(TagNames.RNG)
public class RandomShuffleTest extends BaseNd4jTestWithBackends {

    private static INDArray shuffle(INDArray x) {
        return Nd4j.exec(DynamicCustomOp.builder("random_shuffle").addInputs(x).build())[0];
    }

    private static void shuffleInPlace(INDArray x) {
        Nd4j.exec(DynamicCustomOp.builder("random_shuffle").addInputs(x).addOutputs(x).callInplace(true).build());
    }

    /** Positions 0 .. n - 1 as values: the output's values are the permutation. */
    private static INDArray positions(long n) {
        return Nd4j.linspace(DataType.DOUBLE, 0.0, 1.0, n);
    }

    private static void assertPermutation(INDArray input, INDArray output, String what) {
        assertArrayEquals(input.shape(), output.shape(), what + ": shape");
        double[] in = input.dup('c').data().asDouble();
        double[] out = output.dup('c').data().asDouble();
        Arrays.sort(in);
        Arrays.sort(out);
        assertArrayEquals(in, out, 0.0, what + ": not a permutation of the input");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void vectorsArePermutedReproducibly(Nd4jBackend backend) {
        for (long n : new long[]{2, 3, 1000, 1023, 1025, 5000, (1L << 22) + 5}) {
            INDArray x = positions(n);
            Nd4j.getRandom().setSeed(42);
            INDArray first = shuffle(x);
            assertPermutation(x, first, "length " + n);
            if (n >= 1000)
                assertFalse(first.equals(x), "length " + n + ": left in order");
            Nd4j.getRandom().setSeed(42);
            assertEquals(first, shuffle(x), "length " + n + ": the same seed gave another permutation");
            Nd4j.getRandom().setSeed(42);
            INDArray inPlace = x.dup();
            shuffleInPlace(inPlace);
            assertEquals(first, inPlace, "length " + n + ": in place differs from out of place");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void successiveShufflesDiffer(Nd4jBackend backend) {
        // the generator moves past the numbers a shuffle drew: the next shuffle draws others
        Nd4j.getRandom().setSeed(7);
        INDArray x = positions(1000);
        assertFalse(shuffle(x).equals(shuffle(x)), "vector");
        INDArray rows = positions(150).reshape(50, 3);
        assertFalse(shuffle(rows).equals(shuffle(rows)), "rows");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void rowsMoveWhole(Nd4jBackend backend) {
        for (long rowsCount : new long[]{2, 50, 1025}) {
            INDArray rows = positions(rowsCount * 3).reshape(rowsCount, 3);
            Nd4j.getRandom().setSeed(11);
            INDArray out = shuffle(rows);
            assertPermutation(rows, out, rowsCount + " rows");
            for (long r = 0; r < rowsCount; r++) {
                double first = out.getDouble(r, 0);
                assertEquals(first + 1, out.getDouble(r, 1), 0.0, "row " + r + " split");
                assertEquals(first + 2, out.getDouble(r, 2), 0.0, "row " + r + " split");
            }
            Nd4j.getRandom().setSeed(11);
            INDArray inPlace = rows.dup();
            shuffleInPlace(inPlace);
            assertEquals(out, inPlace, rowsCount + " rows: in place differs from out of place");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void viewsAreShuffledAsTheirCopies(Nd4jBackend backend) {
        INDArray big = positions(4000).reshape(1000, 4);
        // a column ([1000, 1] with a stride of 4) and every second row, both past their buffers' starts
        INDArray column = big.get(NDArrayIndex.all(), NDArrayIndex.interval(2, 3));
        INDArray everySecondRow = big.get(NDArrayIndex.interval(1, 2, 1000), NDArrayIndex.all());
        for (INDArray view : new INDArray[]{column, everySecondRow}) {
            Nd4j.getRandom().setSeed(3);
            INDArray expected = shuffle(view.dup());
            Nd4j.getRandom().setSeed(3);
            assertEquals(expected, shuffle(view), Arrays.toString(view.shape()));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void nothingToShuffle(Nd4jBackend backend) {
        // one element, or one row ([1, N]: TensorFlow shuffles along dimension 0)
        INDArray one = Nd4j.scalar(5.0).reshape(1);
        assertEquals(one, shuffle(one));
        INDArray row = positions(10).reshape(1, 10);
        assertEquals(row, shuffle(row));
    }
}
