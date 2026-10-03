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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.CumProd;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The cumprod gradient against the derivative of each product taken directly. Along one sequence x, output i of the
 * cumulative product multiplies the entries its scan has reached: x[0..i] going forward, x[i..n-1] in reverse, and
 * without x[i] itself when the scan is exclusive. The loss is sum(w * cumprod(x)) with a weight for every output, so
 * dL/dx[k] = sum over outputs i that reach k of w[i] * (the product of the other entries i reaches). Taking the product
 * of the others instead of dividing by x[k] keeps the reference exact for any x.
 *
 * The gradient graph is built from copies of the forward ops that are rebuilt from their serialized form, so the axes
 * and the exclusive and reverse flags of the forward op have to survive that for the backward op to get them.
 */
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class CumProdGradientTest extends BaseNd4jTestWithBackends {

    private static final long[] SHAPE = {5, 4};

    /** Both flags and both axes: each of the eight scans must reach its own gradient. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void everyScanAlongEveryAxis(Nd4jBackend backend) {
        for (int axis = 0; axis < SHAPE.length; axis++) {
            for (boolean exclusive : new boolean[]{false, true}) {
                for (boolean reverse : new boolean[]{false, true}) {
                    check(axis, exclusive, reverse, false);
                }
            }
        }
    }

    /** Without axes the op scans the whole array as one sequence, and its gradient has to do the same. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void scanOverTheWholeArray(Nd4jBackend backend) {
        for (boolean exclusive : new boolean[]{false, true}) {
            for (boolean reverse : new boolean[]{false, true}) {
                check(-1, exclusive, reverse, false);
            }
        }
    }

    /**
     * The scan itself is the loss variable, as in a plain gradient check of the op: the gradient at it is the scalar
     * one, the same at every output.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void scalarGradientAtTheScan(Nd4jBackend backend) {
        for (int axis = 0; axis < SHAPE.length; axis++) {
            for (boolean exclusive : new boolean[]{false, true}) {
                for (boolean reverse : new boolean[]{false, true}) {
                    check(axis, exclusive, reverse, true);
                }
            }
        }
    }

    /**
     * @param axis       the axis scanned along, or -1 to scan the whole array as one sequence
     * @param scalarSeed whether the scan is the loss variable itself (every weight is then one), not weighted
     */
    private static void check(int axis, boolean exclusive, boolean reverse, boolean scalarSeed) {
        double[] x = values(20, 1);
        double[] w = scalarSeed ? ones(20) : values(20, 4);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.createFromArray(x).reshape('c', SHAPE));
        SDVariable scan = axis < 0 ? new CumProd(sd, in, exclusive, reverse).outputVariable()
                : sd.cumprod(in, exclusive, reverse, axis);
        if (scalarSeed) {
            scan.markAsLoss();
        } else {
            SDVariable weights = sd.constant("w", Nd4j.createFromArray(w).reshape('c', SHAPE));
            SDVariable loss = scan.mul(weights).sum();
            loss.markAsLoss();
        }
        Map<String, INDArray> gradients = sd.calculateGradients(Collections.emptyMap(), "in");

        String label = "axis " + (axis < 0 ? "none" : axis) + ", exclusive=" + exclusive + ", reverse=" + reverse
                + (scalarSeed ? ", scalar gradient" : ", weighted");
        assertClose(label, closedForm(x, w, axis, exclusive, reverse), gradients.get("in"));
    }

    /** The gradient of sum(w * cumprod(x)), differentiating every product directly. */
    private static double[] closedForm(double[] x, double[] w, int axis, boolean exclusive, boolean reverse) {
        double[] dx = new double[x.length];
        // A sequence is the elements at base, base + stride, ... base + (n - 1) * stride.
        int n = axis < 0 ? x.length : (int) SHAPE[axis];
        int stride = 1;
        if (axis >= 0) {
            for (int a = axis + 1; a < SHAPE.length; a++) stride *= (int) SHAPE[a];
        }
        int sequences = x.length / n;
        for (int sequence = 0; sequence < sequences; sequence++) {
            int base = axis < 0 ? 0 : (sequence / stride) * n * stride + sequence % stride;
            for (int i = 0; i < n; i++) {
                int first = reverse ? (exclusive ? i + 1 : i) : 0;
                int last = reverse ? n - 1 : (exclusive ? i - 1 : i);
                for (int k = first; k <= last; k++) {
                    double others = 1;
                    for (int j = first; j <= last; j++) {
                        if (j != k) others *= x[base + j * stride];
                    }
                    dx[base + k * stride] += w[base + i * stride] * others;
                }
            }
        }
        return dx;
    }

    /** Non-zero, with |value| in [0.5, 1.6]: the gradient divides by the entries, so zeros are left out. */
    private static double[] values(int count, int seed) {
        double[] values = new double[count];
        for (int i = 0; i < count; i++) {
            double magnitude = 0.5 + ((i * 7 + seed * 13) % 12) / 10.0;
            values[i] = (i + seed) % 3 == 0 ? -magnitude : magnitude;
        }
        return values;
    }

    private static double[] ones(int count) {
        double[] ones = new double[count];
        Arrays.fill(ones, 1.0);
        return ones;
    }

    private static void assertClose(String label, double[] expected, INDArray actual) {
        assertArrayEquals(SHAPE, actual.shape(), label + " shape");
        for (int row = 0; row < SHAPE[0]; row++) {
            for (int column = 0; column < SHAPE[1]; column++) {
                double want = expected[(int) (row * SHAPE[1] + column)];
                assertEquals(want, actual.getDouble(row, column), 1e-10 * (1 + Math.abs(want)),
                        label + " " + Arrays.toString(new int[]{row, column}));
            }
        }
    }
}
