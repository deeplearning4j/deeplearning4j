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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The clipByNorm gradient against its closed form. Each group of elements (a column, a row, or the whole array) with an
 * L2 norm n above the clip value c is scaled to out = x * c / n, so for the loss sum(w * out) the gradient of x is
 * (c / n) * (w - x * dot(w, x) / n^2); a group at or below c passes through, and its gradient is w. The weights differ
 * from element to element: with a uniform w the dot product is w times the sum of x and a gradient that mixes the two
 * up cannot be told from the right one.
 *
 * The columns and rows below take norms from well under to well over the clip value, so some groups of every array are
 * clipped and some are not, and the groups are scattered across the array's memory.
 */
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class ClipByNormGradientTest extends BaseNd4jTestWithBackends {

    private static final long[] SHAPE = {5, 4};
    private static final double CLIP = 2.0;

    /** Norms of the columns, as multiples of the clip value. */
    private static final double[] COLUMN_NORMS = {0.7, 1.3, 0.9, 2.0};
    /** Norms of the rows, as multiples of the clip value. */
    private static final double[] ROW_NORMS = {0.6, 1.4, 0.8, 1.2, 1.7};

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void alongColumns(Nd4jBackend backend) {
        double[] x = scaled(0, COLUMN_NORMS);
        check("columns", x, 0);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void alongRows(Nd4jBackend backend) {
        double[] x = scaled(1, ROW_NORMS);
        check("rows", x, 1);
    }

    /** No dimensions: one norm for the whole array, clipped or not. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void wholeArray(Nd4jBackend backend) {
        for (double norm : new double[]{0.5, 1.5}) {
            double[] x = values(20, 1);
            double scale = norm * CLIP / Math.sqrt(dot(x, x));
            for (int i = 0; i < x.length; i++) x[i] *= scale;
            check("whole array at " + norm + " times the clip value", x, -1);
        }
    }

    // clipByAvgNorm compares the average norm a = n / (elements in the group) with c; the output is x * c / a and the
    // gradient (c / a) * (w - x * dot(w, x) / n^2): the second term divides by the plain norm squared.

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void averagedAlongColumns(Nd4jBackend backend) {
        check("averaged columns", scaled(0, COLUMN_NORMS, true), 0, true);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void averagedAlongRows(Nd4jBackend backend) {
        check("averaged rows", scaled(1, ROW_NORMS, true), 1, true);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void averagedWholeArray(Nd4jBackend backend) {
        for (double norm : new double[]{0.5, 1.5}) {
            double[] x = values(20, 1);
            double scale = norm * CLIP * x.length / Math.sqrt(dot(x, x));
            for (int i = 0; i < x.length; i++) x[i] *= scale;
            check("averaged whole array at " + norm + " times the clip value", x, -1, true);
        }
    }

    /**
     * The forward ops on FLOAT arrays, the clip value given as an argument and (clipbyavgnorm, which takes either) as an
     * input array of another type. The CUDA kernel reads the clip value as the array's type: clipbyavgnorm builds it
     * from its floating point argument as DOUBLE, which a FLOAT kernel read as 0 and so clipped every slice to 0.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void floatForward(Nd4jBackend backend) {
        for (boolean average : new boolean[]{false, true}) {
            for (int dimension = -1; dimension <= 1; dimension++) {
                double[] norms = dimension == 0 ? COLUMN_NORMS : dimension == 1 ? ROW_NORMS : new double[]{1.5};
                double[] x = dimension < 0 ? scaledWhole(1.5, average) : scaled(dimension, norms, average);
                INDArray in = Nd4j.createFromArray(x).reshape('c', SHAPE).castTo(DataType.FLOAT);
                double[] xFloat = in.dup().data().asDouble();
                double[] expected = clippedForward(xFloat, dimension, average);
                long[] dimensions = dimension < 0 ? new long[0] : new long[]{dimension};
                String op = average ? "clipbyavgnorm" : "clipbynorm";
                for (boolean clipAsInput : average ? new boolean[]{false, true} : new boolean[]{false}) {
                    INDArray out = Nd4j.create(DataType.FLOAT, SHAPE);
                    DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder(op).addInputs(in)
                            .addOutputs(out).addIntegerArguments(dimensions);
                    if (clipAsInput) builder.addInputs(Nd4j.scalar(DataType.DOUBLE, CLIP));
                    else builder.addFloatingPointArguments(CLIP);
                    Nd4j.exec(builder.build());
                    String label = op + " dimension " + dimension + (clipAsInput ? ", clip value input" : "");
                    assertClose(label, expected, out, 1e-6);
                }
            }
        }
    }

    /**
     * clipbynorm_bp over more elements than 512 * 1024: the CUDA kernel's launch once swapped its block and thread
     * counts, which asked for more than 1024 threads per block from that size on.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void largeBackprop(Nd4jBackend backend) {
        int rows = 600, columns = 1000;
        double[] x = new double[rows * columns];
        double[] w = new double[rows * columns];
        for (int i = 0; i < x.length; i++) {
            // each row's values times 0.5 to 1.4: row norms from about 18 to 52
            double rowScale = 0.5 + (i / columns % 7) * 0.15;
            x[i] = ((i * 37) % 101 - 50) / 25.0 * rowScale;
            w[i] = ((i * 53) % 89 - 44) / 30.0;
        }
        INDArray in = Nd4j.createFromArray(x).reshape('c', rows, columns);
        INDArray gradOut = Nd4j.createFromArray(w).reshape('c', rows, columns);
        INDArray gradIn = Nd4j.create(DataType.DOUBLE, rows, columns);
        // clip at 30 leaves the rows scaled by 0.5, 0.65 and 0.8 unclipped and clips the others
        double clip = 30.0;
        Nd4j.exec(DynamicCustomOp.builder("clipbynorm_bp").addInputs(in, gradOut).addOutputs(gradIn)
                .addFloatingPointArguments(clip).addIntegerArguments(1L).build());

        double[] actual = gradIn.data().asDouble();
        int clipped = 0;
        for (int row = 0; row < rows; row++) {
            double norm2 = 0, dotWX = 0;
            for (int c = 0; c < columns; c++) {
                int i = row * columns + c;
                norm2 += x[i] * x[i];
                dotWX += w[i] * x[i];
            }
            double norm = Math.sqrt(norm2);
            if (norm > clip) clipped++;
            for (int c = 0; c < columns; c++) {
                int i = row * columns + c;
                double want = norm > clip ? clip / norm * (w[i] - x[i] * dotWX / norm2) : w[i];
                assertEquals(want, actual[i], 1e-10 * (1 + Math.abs(want)), "row " + row + " column " + c);
            }
        }
        assertTrue(clipped > 0 && clipped < rows, "clipped rows: " + clipped);
    }

    /** The output of clipByNorm (or clipByAvgNorm) of x, with the slices along the dimension (or the whole array). */
    private static double[] clippedForward(double[] x, int dimension, boolean average) {
        double[] out = new double[x.length];
        int groups = dimension < 0 ? 1 : (int) SHAPE[1 - dimension];
        for (int group = 0; group < groups; group++) {
            double[] gx = slice(x, dimension, group);
            double norm = Math.sqrt(dot(gx, gx));
            double compared = average ? norm / gx.length : norm;
            double[] clipped = new double[gx.length];
            for (int i = 0; i < gx.length; i++) clipped[i] = compared > CLIP ? gx[i] * CLIP / compared : gx[i];
            store(out, dimension, group, clipped);
        }
        return out;
    }

    /** Values whose whole-array (average) norm is `norm` times the clip value. */
    private static double[] scaledWhole(double norm, boolean average) {
        double[] x = values(20, 1);
        double scale = norm * CLIP * (average ? x.length : 1) / Math.sqrt(dot(x, x));
        for (int i = 0; i < x.length; i++) x[i] *= scale;
        return x;
    }

    /** @param dimension the dimension whose slices are clipped, or -1 for no dimension */
    private static void check(String label, double[] x, int dimension) {
        check(label, x, dimension, false);
    }

    /** @param average whether the group's average norm (clipByAvgNorm) is compared with the clip value */
    private static void check(String label, double[] x, int dimension, boolean average) {
        double[] w = values(20, 4);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.createFromArray(x).reshape('c', SHAPE));
        long[] dimensions = dimension < 0 ? new long[0] : new long[]{dimension};
        SDVariable clipped = average ? sd.math().clipByAvgNorm(in, CLIP, dimensions)
                : sd.math().clipByNorm(in, CLIP, dimensions);
        SDVariable weights = sd.constant("w", Nd4j.createFromArray(w).reshape('c', SHAPE));
        SDVariable loss = clipped.mul(weights).sum();
        loss.markAsLoss();
        Map<String, INDArray> gradients = sd.calculateGradients(Collections.emptyMap(), "in");

        assertClose(label, closedForm(x, w, dimension, average), gradients.get("in"));
    }

    /** The gradient of sum(w * clipByNorm(x)) with the slices along the dimension, or the whole array, as the groups. */
    private static double[] closedForm(double[] x, double[] w, int dimension, boolean average) {
        double[] dx = new double[x.length];
        int groups = dimension < 0 ? 1 : (int) SHAPE[1 - dimension];
        for (int group = 0; group < groups; group++) {
            double[] gx = slice(x, dimension, group);
            double[] gw = slice(w, dimension, group);
            double norm = Math.sqrt(dot(gx, gx));
            double compared = average ? norm / gx.length : norm;
            double dotWX = dot(gw, gx);
            double[] gradient = new double[gx.length];
            for (int i = 0; i < gx.length; i++) {
                gradient[i] = compared > CLIP ? CLIP / compared * (gw[i] - gx[i] * dotWX / (norm * norm)) : gw[i];
            }
            store(dx, dimension, group, gradient);
        }
        return dx;
    }

    /** The elements of one slice along the dimension in the order of the other axis: the column (row) at `group`. */
    private static double[] slice(double[] values, int dimension, int group) {
        if (dimension < 0) return values.clone();
        int length = (int) SHAPE[dimension];
        double[] slice = new double[length];
        for (int i = 0; i < length; i++) slice[i] = values[position(dimension, group, i)];
        return slice;
    }

    private static void store(double[] target, int dimension, int group, double[] slice) {
        if (dimension < 0) {
            System.arraycopy(slice, 0, target, 0, slice.length);
            return;
        }
        for (int i = 0; i < slice.length; i++) target[position(dimension, group, i)] = slice[i];
    }

    /** The flat C-order position of element i of the slice at `group`: dimension 0 runs down a column, 1 along a row. */
    private static int position(int dimension, int group, int i) {
        return dimension == 0 ? i * (int) SHAPE[1] + group : group * (int) SHAPE[1] + i;
    }

    /** Values whose slices along the dimension have the given norms, as multiples of the clip value. */
    private static double[] scaled(int dimension, double[] norms) {
        return scaled(dimension, norms, false);
    }

    /** As above, with the average norm (the norm over the slice length) at those multiples when `average` is set. */
    private static double[] scaled(int dimension, double[] norms, boolean average) {
        double[] x = values(20, 1);
        for (int group = 0; group < norms.length; group++) {
            double[] slice = slice(x, dimension, group);
            double scale = norms[group] * CLIP * (average ? slice.length : 1) / Math.sqrt(dot(slice, slice));
            for (int i = 0; i < slice.length; i++) slice[i] *= scale;
            store(x, dimension, group, slice);
        }
        return x;
    }

    private static double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    /** Neither zero nor repeating quickly: |value| in [0.4, 1.6], alternating in sign. */
    private static double[] values(int count, int seed) {
        double[] values = new double[count];
        for (int i = 0; i < count; i++) {
            double magnitude = 0.4 + ((i * 7 + seed * 13) % 13) / 10.0;
            values[i] = (i + seed) % 2 == 0 ? magnitude : -magnitude;
        }
        return values;
    }

    private static void assertClose(String label, double[] expected, INDArray actual) {
        assertClose(label, expected, actual, 1e-10);
    }

    private static void assertClose(String label, double[] expected, INDArray actual, double relativeTolerance) {
        assertArrayEquals(SHAPE, actual.shape(), label + " shape");
        for (int row = 0; row < SHAPE[0]; row++) {
            for (int column = 0; column < SHAPE[1]; column++) {
                double want = expected[(int) (row * SHAPE[1] + column)];
                assertEquals(want, actual.getDouble(row, column), relativeTolerance * (1 + Math.abs(want)),
                        label + " " + Arrays.toString(new int[]{row, column}));
            }
        }
    }
}
