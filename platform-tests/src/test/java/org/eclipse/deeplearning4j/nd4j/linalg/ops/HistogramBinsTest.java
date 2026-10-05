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
import org.nd4j.linalg.api.ops.impl.transforms.Histogram;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * histogram counts the elements of its input in numBins bins of equal width between the input's smallest and largest
 * value: bin = floor((x - min) / width), width = (max - min) / numBins, the largest value closing the last bin.
 * <p>
 * The CUDA helper gave each block numBins * 256 bytes of dynamic shared memory for counters of numBins * 8 bytes
 * (an element count used as bytes), so every histogram of 192 bins or more failed to launch. Both backends took the
 * bin width in the input's type: an integer input truncated it (0 to 10 over 4 bins is 2.5 wide, not 2), and
 * equal values divided by a width of zero. The CPU helper counted with a SIMD pragma over bins[idx]++ (a vectorized
 * scatter loses counts), read the input as dense whatever its strides, and counted in int.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class HistogramBinsTest extends BaseNd4jTestWithBackends {

    @Override
    public long getTimeoutMilliseconds() {
        return 600000;
    }

    /** The bin of a value, as the op defines it. */
    private static int bin(double value, double low, double width, int numBins) {
        if (!(width > 0.0))
            return 0;
        double position = (value - low) / width;
        if (!(position > 0.0))
            return 0;
        if (position >= (double) (numBins - 1))
            return numBins - 1;
        return (int) position;
    }

    private static long[] reference(double[] x, int numBins) {
        double low = Double.POSITIVE_INFINITY;
        double high = Double.NEGATIVE_INFINITY;
        for (double v : x) {
            low = Math.min(low, v);
            high = Math.max(high, v);
        }
        double width = (high - low) / (double) numBins;
        long[] counts = new long[numBins];
        for (double v : x)
            counts[bin(v, low, width, numBins)]++;
        return counts;
    }

    private static double[] toDoubles(INDArray a) {
        return a.dup('c').data().asDouble();
    }

    private static INDArray histogram(INDArray input, int numBins) {
        INDArray out = Nd4j.exec(new Histogram(input, numBins))[0];
        Nd4j.getExecutioner().commit();
        assertEquals(DataType.INT64, out.dataType(), "the histogram's type");
        assertArrayEquals(new long[]{numBins}, out.shape(), "the histogram's shape");
        return out;
    }

    private static long[] counts(INDArray input, int numBins) {
        return histogram(input, numBins).dup('c').data().asLong();
    }

    private static double[] gaussian(int n, long seed, double mean, double std) {
        Random r = new Random(seed);
        double[] v = new double[n];
        for (int i = 0; i < n; i++)
            v[i] = mean + std * r.nextGaussian();
        return v;
    }

    private static long sum(long[] a) {
        long s = 0;
        for (long v : a)
            s += v;
        return s;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void integerInputsKeepTheFractionOfTheBinWidth(Nd4jBackend backend) {
        double[] zeroToTen = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        for (DataType type : new DataType[]{DataType.INT32, DataType.INT64, DataType.INT8, DataType.UINT8,
                DataType.INT16, DataType.DOUBLE, DataType.FLOAT}) {
            INDArray x = Nd4j.createFromArray(zeroToTen).castTo(type);
            // 2.5 wide: 0 1 2 | 3 4 | 5 6 7 | 8 9 10
            assertArrayEquals(new long[]{3, 2, 3, 3}, counts(x, 4), type + ", 4 bins");
            // 3.33 wide: 0 1 2 3 | 4 5 6 | 7 8 9 10
            assertArrayEquals(new long[]{4, 3, 4}, counts(x, 3), type + ", 3 bins");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void everyNumberOfBinsCountsLikeTheReference(Nd4jBackend backend) {
        // 191 bins fit the old launch, 192 and more did not; 20000 do not fit a block's shared memory
        int[] binCounts = {1, 2, 20, 191, 192, 1000, 10000, 20000, 100000};
        for (int n : new int[]{1, 100, 100000, 1 << 20}) {
            double[] floats = gaussian(n, 11L * n, 3.0, 2.0);
            double[] spread = new double[n];
            Random r = new Random(5L * n);
            for (int i = 0; i < n; i++)
                spread[i] = r.nextInt(5000) - 1000;
            for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.INT32, DataType.INT64}) {
                INDArray x = Nd4j.createFromArray(type.isFPType() ? floats : spread).castTo(type);
                double[] seen = toDoubles(x);
                for (int numBins : binCounts) {
                    long[] expected = reference(seen, numBins);
                    long[] actual = counts(x, numBins);
                    assertEquals(n, sum(actual), type + ", " + n + " elements, " + numBins + " bins: counted elements");
                    assertArrayEquals(expected, actual, type + ", " + n + " elements, " + numBins + " bins");
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void halfPrecisionInputsAreBinnedByTheirValues(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT16, DataType.BFLOAT16}) {
            INDArray x = Nd4j.createFromArray(gaussian(5000, 3, 0.5, 1.5)).castTo(type);
            double[] seen = toDoubles(x);
            for (int numBins : new int[]{20, 192, 1000}) {
                assertArrayEquals(reference(seen, numBins), counts(x, numBins), type + ", " + numBins + " bins");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void equalValuesFallInTheFirstBin(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.INT32, DataType.INT64}) {
            for (int n : new int[]{1, 1000, 100000}) {
                INDArray x = Nd4j.valueArrayOf(new long[]{n}, 7.0, type);
                for (int numBins : new int[]{1, 4, 50, 192, 1000}) {
                    long[] expected = new long[numBins];
                    expected[0] = n;
                    assertArrayEquals(expected, counts(x, numBins), type + ", " + n + " equal values, " + numBins + " bins");
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void viewsAreCountedThroughTheirStrides(Nd4jBackend backend) {
        int rows = 300;
        int cols = 500;
        INDArray big = Nd4j.createFromArray(gaussian(rows * cols, 21, 0.0, 10.0)).reshape('c', rows, cols);
        INDArray bigF = big.dup('f');
        INDArray[] views = {
                // a column: stride 500
                big.get(NDArrayIndex.all(), NDArrayIndex.interval(7, 8)),
                // every third element of every second row
                big.get(NDArrayIndex.interval(1, 2, rows), NDArrayIndex.interval(0, 3, cols)),
                // a block that starts in the middle of the buffer
                big.get(NDArrayIndex.interval(100, 250), NDArrayIndex.interval(50, 450)),
                // one row, past the buffer's start
                big.getRow(123),
                // F order, whole and as a stepped block
                bigF,
                bigF.get(NDArrayIndex.interval(0, 2, rows), NDArrayIndex.interval(10, 490)),
                // permuted
                big.reshape('c', 30, 10, 500).permute(2, 0, 1)};
        for (int v = 0; v < views.length; v++) {
            double[] seen = toDoubles(views[v]);
            for (int numBins : new int[]{10, 192, 1000}) {
                assertArrayEquals(reference(seen, numBins), counts(views[v], numBins),
                        "view " + v + ", " + numBins + " bins");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void integerOutputsOfAnyTypeAndViewsAreWritten(Nd4jBackend backend) {
        double[] values = gaussian(20000, 8, 0.0, 3.0);
        INDArray x = Nd4j.createFromArray(values);
        for (int numBins : new int[]{20, 192, 1000}) {
            long[] expected = reference(values, numBins);
            double[] expectedDoubles = new double[numBins];
            for (int b = 0; b < numBins; b++)
                expectedDoubles[b] = expected[b];

            for (DataType type : new DataType[]{DataType.INT32, DataType.INT64, DataType.INT16, DataType.UINT16}) {
                INDArray out = Nd4j.zeros(type, numBins);
                Nd4j.exec(new Histogram(x, out));
                Nd4j.getExecutioner().commit();
                assertArrayEquals(expectedDoubles, toDoubles(out), 0.0, type + " output, " + numBins + " bins");
            }

            // an output that is a column of a wider array: its neighbours stay as they were
            INDArray wide = Nd4j.zeros(DataType.INT64, numBins, 3).assign(-5);
            INDArray column = wide.get(NDArrayIndex.all(), NDArrayIndex.point(1));
            Nd4j.exec(new Histogram(x, column));
            Nd4j.getExecutioner().commit();
            assertArrayEquals(expectedDoubles, toDoubles(wide.get(NDArrayIndex.all(), NDArrayIndex.point(1))), 0.0,
                    "output column, " + numBins + " bins");
            for (int c : new int[]{0, 2}) {
                double[] around = toDoubles(wide.get(NDArrayIndex.all(), NDArrayIndex.point(c)));
                for (int b = 0; b < numBins; b++)
                    assertEquals(-5.0, around[b], 0.0, "column " + c + " bin " + b + " changed");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void repeatedRunsAgreeAndLeaveTheInputAlone(Nd4jBackend backend) {
        INDArray x = Nd4j.createFromArray(gaussian(300000, 4, 1.0, 4.0)).castTo(DataType.FLOAT);
        INDArray before = x.dup();
        long[] first = counts(x, 1000);
        for (int run = 0; run < 3; run++)
            assertArrayEquals(first, counts(x, 1000), "run " + run);
        assertEquals(before, x, "the input changed");
        assertArrayEquals(reference(toDoubles(x), 1000), first);
    }
}
