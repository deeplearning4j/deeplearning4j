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

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * cumsum and cumprod against a plain Java scan. Without axes the whole array is one sequence in C order; with an axis
 * every vector along it is scanned on its own. `exclusive` leaves each element out of its own result, `reverse` scans
 * from the end. On CUDA the scan once ran as a host loop (invisible to a captured CUDA graph); it runs on the device
 * now, chunk by chunk, so long sequences, strided views, 'f' order and in-place execution are covered here. All values
 * keep every partial result exactly representable, so CPU and device agree exactly.
 */
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@NativeTag
public class CumulativeScanTest extends BaseNd4jTestWithBackends {

    private static INDArray scan(String op, INDArray in, INDArray out, boolean exclusive, boolean reverse,
                                 long... axes) {
        // integer arguments: [exclusive, reverse, axes...]
        long[] arguments = new long[2 + axes.length];
        arguments[0] = exclusive ? 1 : 0;
        arguments[1] = reverse ? 1 : 0;
        System.arraycopy(axes, 0, arguments, 2, axes.length);
        Nd4j.exec(DynamicCustomOp.builder(op).addInputs(in).addOutputs(out).addIntegerArguments(arguments).build());
        return out;
    }

    /** The scan of one sequence. */
    private static double[] reference(double[] values, boolean product, boolean exclusive, boolean reverse) {
        double[] out = new double[values.length];
        double running = product ? 1 : 0;
        for (int step = 0; step < values.length; step++) {
            int i = reverse ? values.length - 1 - step : step;
            double next = product ? running * values[i] : running + values[i];
            out[i] = exclusive ? running : next;
            running = next;
        }
        return out;
    }

    /** The scan of a [rows, cols] matrix along `axis` (0: down the columns, 1: along the rows), or flat when -1. */
    private static INDArray reference(INDArray matrix, boolean product, boolean exclusive, boolean reverse, int axis) {
        int rows = (int) matrix.size(0), cols = (int) matrix.size(1);
        double[][] m = matrix.toDoubleMatrix();
        double[][] out = new double[rows][cols];
        if (axis < 0) {
            double[] flat = new double[rows * cols];
            for (int r = 0; r < rows; r++) System.arraycopy(m[r], 0, flat, r * cols, cols);
            double[] scanned = reference(flat, product, exclusive, reverse);
            for (int r = 0; r < rows; r++) System.arraycopy(scanned, r * cols, out[r], 0, cols);
        } else if (axis == 1) {
            for (int r = 0; r < rows; r++) out[r] = reference(m[r], product, exclusive, reverse);
        } else {
            for (int c = 0; c < cols; c++) {
                double[] column = new double[rows];
                for (int r = 0; r < rows; r++) column[r] = m[r][c];
                double[] scanned = reference(column, product, exclusive, reverse);
                for (int r = 0; r < rows; r++) out[r][c] = scanned[r];
            }
        }
        return Nd4j.createFromArray(out);
    }

    private static INDArray matrix(char order) {
        double[] values = {1, 2, -1, 0.5, 3, -2, 1.5, 2, -0.5, 4, 1, -1};
        return Nd4j.create(values, new long[]{3, 4}, order);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void everyAxisAndDirection(Nd4jBackend backend) {
        for (char order : new char[]{'c', 'f'}) {
            for (String op : new String[]{"cumsum", "cumprod"}) {
                for (boolean exclusive : new boolean[]{false, true}) {
                    for (boolean reverse : new boolean[]{false, true}) {
                        for (int axis = -1; axis <= 1; axis++) {
                            INDArray in = matrix(order);
                            INDArray out = Nd4j.create(DataType.DOUBLE, 3, 4);
                            if (axis < 0) scan(op, in, out, exclusive, reverse);
                            else scan(op, in, out, exclusive, reverse, axis);
                            assertEquals(reference(in, op.equals("cumprod"), exclusive, reverse, axis), out,
                                    op + " order=" + order + " exclusive=" + exclusive + " reverse=" + reverse
                                            + " axis=" + axis);
                        }
                    }
                }
            }
        }
    }

    /** Far longer than a block: the carry runs across many chunks. Every partial sum is an exact integer. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void longSequence(Nd4jBackend backend) {
        int n = 100_000;
        INDArray ones = Nd4j.ones(DataType.DOUBLE, n);
        for (boolean exclusive : new boolean[]{false, true}) {
            for (boolean reverse : new boolean[]{false, true}) {
                INDArray out = scan("cumsum", ones, Nd4j.create(DataType.DOUBLE, n), exclusive, reverse);
                double[] expected = reference(ones.toDoubleVector(), false, exclusive, reverse);
                assertEquals(Nd4j.createFromArray(expected), out, "exclusive=" + exclusive + " reverse=" + reverse);
            }
        }
    }

    /** A column of a matrix is a strided view with an offset; the rest of the matrix stays as it was. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void stridedViewAndInPlace(Nd4jBackend backend) {
        INDArray matrix = matrix('c');
        INDArray column = matrix.getColumn(2);
        double[] expected = reference(column.toDoubleVector(), false, false, false);
        INDArray out = scan("cumsum", column, Nd4j.create(DataType.DOUBLE, 3), false, false);
        assertEquals(Nd4j.createFromArray(expected), out);

        // in place on the view: only that column changes
        INDArray before = matrix.dup();
        scan("cumsum", column, column, false, false);
        before.putColumn(2, Nd4j.createFromArray(expected));
        assertEquals(before, matrix);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
