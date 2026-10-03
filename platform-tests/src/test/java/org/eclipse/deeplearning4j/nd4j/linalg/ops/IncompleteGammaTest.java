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

import org.apache.commons.math3.special.Gamma;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.custom.Igamma;
import org.nd4j.linalg.api.ops.custom.Igammac;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * igamma and igammac are the regularized incomplete gamma functions P(a, x) and Q(a, x) = 1 - P(a, x)
 * (TensorFlow's Igamma and Igammac). The values come from Apache Commons Math. The grid reaches x far
 * beyond 10, where the native series once stopped after a dozen terms, and Q far below 1e-16, where
 * 1 - P cancels to 0.
 */
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@NativeTag
public class IncompleteGammaTest extends BaseNd4jTestWithBackends {

    /** 5e-4 is below 0.001, where Gamma(a) was once approximated as 1 / (a (1 + 0.5772 a)), 1.6e-7 off there. */
    private static final double[] SHAPES = {1e-7, 5e-4, 0.01, 0.5, 1.0, 2.5, 10.0, 47.5, 300.0, 1000.0};
    private static final double[] POINTS = {1e-3, 0.5, 1.0, 2.0, 9.9, 10.0, 30.0, 60.0, 100.0, 450.0, 1100.0, 5000.0};

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void matchesCommonsMathInDouble(Nd4jBackend backend) {
        compareGrid(DataType.DOUBLE, 1e-10);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void matchesCommonsMathInFloat(Nd4jBackend backend) {
        compareGrid(DataType.FLOAT, 4e-7);
    }

    /** a <= 0 and x < 0 are domain errors and NaN propagates; P(a, 0) = 0 and Q(a, 0) = 1. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void domainEdges(Nd4jBackend backend) {
        double[] a = {0.0, -1.0, 2.0, Double.NaN, 2.0, 3.0};
        double[] x = {1.0, 1.0, -0.5, 1.0, Double.NaN, 0.0};
        double[] p = run(DataType.DOUBLE, a, x, false);
        double[] q = run(DataType.DOUBLE, a, x, true);
        for (int i = 0; i < 5; i++) {
            assertTrue(Double.isNaN(p[i]), "igamma(" + a[i] + ", " + x[i] + ") = " + p[i]);
            assertTrue(Double.isNaN(q[i]), "igammac(" + a[i] + ", " + x[i] + ") = " + q[i]);
        }
        assertEquals(0.0, p[5], 0.0);
        assertEquals(1.0, q[5], 0.0);
    }

    /** a [3, 1] against x [1, 4] gives every pair. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void broadcastsShapeAgainstPoint(Nd4jBackend backend) {
        double[] a = {0.5, 4.0, 20.0};
        double[] x = {0.25, 3.0, 15.0, 80.0};
        INDArray out = Nd4j.exec(new Igamma(Nd4j.createFromArray(a).reshape(3, 1), Nd4j.createFromArray(x).reshape(1, 4)))[0];
        assertArrayEquals(new long[]{3, 4}, out.shape());
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < x.length; j++) {
                double expected = Gamma.regularizedGammaP(a[i], x[j], 1e-16, Integer.MAX_VALUE);
                assertEquals(expected, out.getDouble(i, j), 1e-10 * expected, "igamma(" + a[i] + ", " + x[j] + ")");
            }
        }
    }

    private static void compareGrid(DataType type, double relativeTolerance) {
        int n = SHAPES.length * POINTS.length;
        double[] a = new double[n];
        double[] x = new double[n];
        for (int i = 0; i < SHAPES.length; i++) {
            for (int j = 0; j < POINTS.length; j++) {
                a[i * POINTS.length + j] = typed(type, SHAPES[i]);
                x[i * POINTS.length + j] = typed(type, POINTS[j]);
            }
        }
        double[] p = run(type, a, x, false);
        double[] q = run(type, a, x, true);
        double smallest = type == DataType.FLOAT ? Float.MIN_NORMAL : Double.MIN_NORMAL;
        // Where either side forms Q as 1 - P (or P as 1 - Q) the subtraction cancels: neither can be closer than a
        // few ulps of 1 there, however small the result. Commons Math complements Q below x = a + 1 and P from there
        // on; the native code complements Q where x < 1 or x < a and P where x > 1 and x > a. Commons Math's own P
        // is ~13 ulps of 1 off at a = 1e-7 (its scale goes through log Gamma(a), about -log a; Q(1e-7, 0.5) is
        // 5.5977362411e-8, which the native code gets within 1.5 ulps and Commons Math within 13), so the floor
        // allows for both. Where Q is far below 1e-16, no complement is involved and the relative tolerance alone
        // applies.
        double complementFloor = 32 * Math.ulp(1.0);
        for (int k = 0; k < n; k++) {
            double expectedP = Gamma.regularizedGammaP(a[k], x[k], 1e-16, Integer.MAX_VALUE);
            double expectedQ = Gamma.regularizedGammaQ(a[k], x[k], 1e-16, Integer.MAX_VALUE);
            boolean qComplemented = x[k] < a[k] + 1;
            boolean pComplemented = x[k] >= a[k] + 1 || (x[k] > 1 && x[k] > a[k]);
            assertEquals(expectedP, p[k], relativeTolerance * expectedP + (pComplemented ? complementFloor : 0) + smallest,
                    type + " igamma(" + a[k] + ", " + x[k] + ")");
            assertEquals(expectedQ, q[k], relativeTolerance * expectedQ + (qComplemented ? complementFloor : 0) + smallest,
                    type + " igammac(" + a[k] + ", " + x[k] + ")");
        }
    }

    private static double[] run(DataType type, double[] a, double[] x, boolean upper) {
        INDArray shapes = Nd4j.createFromArray(a).castTo(type);
        INDArray points = Nd4j.createFromArray(x).castTo(type);
        INDArray out = Nd4j.exec(upper ? new Igammac(shapes, points) : new Igamma(shapes, points))[0];
        assertEquals(type, out.dataType());
        return out.data().asDouble();
    }

    private static double typed(DataType type, double value) {
        return type == DataType.FLOAT ? (float) value : value;
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
