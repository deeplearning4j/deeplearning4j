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
import org.nd4j.linalg.api.ops.impl.transforms.strict.Erf;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Erfc;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * erf and erfc against reference values, not against another backend. The values are erf and erfc computed to 113 bits
 * (the Kummer series e^(-x^2) 2x / sqrt(pi) sum (2x^2)^n / (2n+1)!! for erf, the Laplace continued fraction for erfc
 * from 2.5 on, 1 - erf below) and rounded to double; the FLOAT cases use the reference of the float-rounded input.
 *
 * The inputs sit on both sides of the branches of the fdlibm algorithm the native backends and the Vulkan expansion
 * share (0.84375, 1.25, 1/0.35, 6) and run the tails of erfc down to 2e-307, where 1 - erf is 0 and where a
 * polynomial fit of erf alone (a 1e-7 absolute error such as Abramowitz-Stegun 7.1.26) is wrong in every digit.
 *
 * Tolerances are relative to the expected value. DOUBLE 1e-14 is 45 ulp: libm's erf and erfc are within 1 ulp, CUDA
 * within a few (its documentation allows 2 for erf and up to 5 for erfc), the Vulkan expansion measures 1 and 2.
 * FLOAT 3e-6 is 25 ulp: erff and erfcf are within 1 ulp on the CPU, a few on CUDA, the Vulkan expansion within 2 on a
 * device that divides with IEEE accuracy and a few more where float division is only the 2.5 ulp Vulkan requires.
 */
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@NativeTag
public class ErfErfcAccuracyTest extends BaseNd4jTestWithBackends {

    private static final double DOUBLE_TOLERANCE = 1e-14;
    private static final double FLOAT_TOLERANCE = 3e-6;
    /** erf and erfc lie in [-1, 2]: an output that was never written still holds this. */
    private static final double SENTINEL = -777.0;

    private static final double[] DOUBLE_X = {
        1e-10, 1e-05, 0.01, 0.1,
        0.25, 0.5, 0.84375, 0.9,
        1.0, 1.25, 1.5, 2.0,
        2.857142857142857, 3.0, 4.0, 5.0,
        6.0, 7.0, 9.0, 12.0,
        20.0, 26.0, 26.5, -1e-05,
        -0.1, -0.5, -0.84375, -1.0,
        -1.25, -2.0, -2.857142857142857, -3.0,
        -5.0, -6.0, -10.0, -26.0};

    private static final double[] DOUBLE_ERF = {
        1.1283791670955126e-10, 1.1283791670579e-05, 0.011283415555849618,
        0.1124629160182849, 0.27632639016823696, 0.5204998778130465,
        0.7672256612323416, 0.7969082124228322, 0.8427007929497149,
        0.9229001282564583, 0.9661051464753108, 0.9953222650189527,
        0.9999466876886117, 0.9999779095030014, 0.9999999845827421,
        0.9999999999984626, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, -1.1283791670579e-05,
        -0.1124629160182849, -0.5204998778130465, -0.7672256612323416,
        -0.8427007929497149, -0.9229001282564583, -0.9953222650189527,
        -0.9999466876886117, -0.9999779095030014, -0.9999999999984626,
        -1.0, -1.0, -1.0};

    private static final double[] DOUBLE_ERFC = {
        0.999999999887162, 0.9999887162083294, 0.9887165844441503,
        0.887537083981715, 0.7236736098317631, 0.4795001221869535,
        0.23277433876765838, 0.20309178757716786, 0.15729920705028513,
        0.07709987174354177, 0.033894853524689274, 0.004677734981047266,
        5.3312311388322795e-05, 2.209049699858544e-05, 1.541725790028002e-08,
        1.537459794428035e-12, 2.1519736712498913e-17, 4.183825607779414e-23,
        4.13703174651381e-37, 1.3562611692059042e-64, 5.395865611607901e-176,
        5.663192408856143e-296, 2.2109076642637343e-307, 1.0000112837916706,
        1.1124629160182848, 1.5204998778130465, 1.7672256612323416,
        1.8427007929497148, 1.9229001282564582, 1.9953222650189528,
        1.9999466876886116, 1.9999779095030015, 1.9999999999984626,
        2.0, 2.0, 2.0};

    private static final float[] FLOAT_X = {
        1e-05f, 0.01f, 0.1f, 0.25f, 0.5f, 0.84375f,
        0.9f, 1f, 1.25f, 1.5f, 2f, 2.857143f,
        3f, 4f, 5f, 6f, 7f, 8f,
        9f, -1e-05f, -0.1f, -0.5f, -0.84375f, -1f,
        -1.25f, -2f, -3f, -5f, -6f, -8f};

    /** erf of the float-rounded FLOAT_X, 9f and beyond being 1 in float */
    private static final double[] FLOAT_ERF = {
        1.1283791385526445e-05, 0.011283415303662439, 0.11246291768297051,
        0.27632639016823696, 0.5204998778130465, 0.7672256612323416,
        0.7969082004549685, 0.8427007929497149, 0.9229001282564583,
        0.9661051464753108, 0.9953222650189527, 0.9999466877105128,
        0.9999779095030014, 0.9999999845827421, 0.9999999999984626,
        1.0, 1.0, 1.0,
        1.0, -1.1283791385526445e-05, -0.11246291768297051,
        -0.5204998778130465, -0.7672256612323416, -0.8427007929497149,
        -0.9229001282564583, -0.9953222650189527, -0.9999779095030014,
        -0.9999999999984626, -1.0, -1.0};

    /** erfc of the float-rounded FLOAT_X; 9f is 4.1e-37, still a normal float */
    private static final double[] FLOAT_ERFC = {
        0.9999887162086145, 0.9887165846963376, 0.8875370823170295,
        0.7236736098317631, 0.4795001221869535, 0.23277433876765838,
        0.20309179954503154, 0.15729920705028513, 0.07709987174354177,
        0.033894853524689274, 0.004677734981047266, 5.331228948722176e-05,
        2.209049699858544e-05, 1.541725790028002e-08, 1.537459794428035e-12,
        2.1519736712498913e-17, 4.183825607779414e-23, 1.1224297172982926e-29,
        4.13703174651381e-37, 1.0000112837913855, 1.1124629176829706,
        1.5204998778130465, 1.7672256612323416, 1.8427007929497148,
        1.9229001282564582, 1.9953222650189528, 1.9999779095030015,
        1.9999999999984626, 2.0, 2.0};

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void erfMatchesReferenceInDouble(Nd4jBackend backend) {
        INDArray x = Nd4j.createFromArray(DOUBLE_X);
        assertRelative("erf", DataType.DOUBLE, DOUBLE_X, DOUBLE_ERF, DOUBLE_TOLERANCE, apply(false, x));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void erfcMatchesReferenceInDouble(Nd4jBackend backend) {
        INDArray x = Nd4j.createFromArray(DOUBLE_X);
        assertRelative("erfc", DataType.DOUBLE, DOUBLE_X, DOUBLE_ERFC, DOUBLE_TOLERANCE, apply(true, x));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void erfMatchesReferenceInFloat(Nd4jBackend backend) {
        INDArray x = Nd4j.createFromArray(FLOAT_X);
        assertRelative("erf", DataType.FLOAT, widen(FLOAT_X), FLOAT_ERF, FLOAT_TOLERANCE, apply(false, x));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void erfcMatchesReferenceInFloat(Nd4jBackend backend) {
        INDArray x = Nd4j.createFromArray(FLOAT_X);
        assertRelative("erfc", DataType.FLOAT, widen(FLOAT_X), FLOAT_ERFC, FLOAT_TOLERANCE, apply(true, x));
    }

    /**
     * erf(+-inf) = +-1, erfc(+inf) = 0 and erfc(-inf) = 2, NaN stays NaN, erf(+-0) = +-0 and erfc(0) = 1, and a
     * huge argument saturates instead of overflowing inside the tail (x^2 is inf, 1 / x^2 is 0, exp underflows).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void specialValuesInDouble(Nd4jBackend backend) {
        double[] x = {Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NaN, 0.0, -0.0, 1e300, -1e300, 30.0, -30.0};
        double[] erf = {1.0, -1.0, Double.NaN, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0};
        double[] erfc = {0.0, 2.0, Double.NaN, 1.0, 1.0, 0.0, 2.0, 0.0, 2.0};
        assertSpecial("erf", DataType.DOUBLE, x, erf, apply(false, Nd4j.createFromArray(x)));
        assertSpecial("erfc", DataType.DOUBLE, x, erfc, apply(true, Nd4j.createFromArray(x)));
    }

    /** The same in FLOAT; erfc(12) is 1.4e-63, far below the smallest float. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void specialValuesInFloat(Nd4jBackend backend) {
        float[] x = {Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN, 0f, -0f, 1e30f, -1e30f, 12f, -12f};
        double[] erf = {1.0, -1.0, Double.NaN, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0};
        double[] erfc = {0.0, 2.0, Double.NaN, 1.0, 1.0, 0.0, 2.0, 0.0, 2.0};
        assertSpecial("erf", DataType.FLOAT, widen(x), erf, apply(false, Nd4j.createFromArray(x)));
        assertSpecial("erfc", DataType.FLOAT, widen(x), erfc, apply(true, Nd4j.createFromArray(x)));
    }

    /**
     * A transposed (permuted, non-contiguous) view is read element by element through its own strides. Twelve of
     * the inputs above: 1e-5, 0.84375, 1.25, 1/0.35, 5, 6, 9, 26, 26.5 and three negative ones (-1e-5, -1.25, -6).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void transposedViewInput(Nd4jBackend backend) {
        int rows = 3;
        int columns = 4;
        int[] picked = {1, 6, 9, 12, 15, 16, 18, 21, 22, 23, 28, 33};
        double[] x = new double[rows * columns];
        double[] erf = new double[rows * columns];
        double[] erfc = new double[rows * columns];
        for (int k = 0; k < picked.length; k++) {
            x[k] = DOUBLE_X[picked[k]];
            erf[k] = DOUBLE_ERF[picked[k]];
            erfc[k] = DOUBLE_ERFC[picked[k]];
        }
        INDArray view = Nd4j.createFromArray(x).reshape(rows, columns).transpose();
        assertArrayEquals(new long[]{columns, rows}, view.shape());
        INDArray erfOut = apply(false, view);
        INDArray erfcOut = apply(true, view);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                int k = i * columns + j;
                assertEquals(erf[k], erfOut.getDouble(j, i), DOUBLE_TOLERANCE * Math.abs(erf[k]),
                        "erf of the transposed view at [" + j + ", " + i + "], x = " + x[k]);
                assertEquals(erfc[k], erfcOut.getDouble(j, i), DOUBLE_TOLERANCE * Math.abs(erfc[k]),
                        "erfc of the transposed view at [" + j + ", " + i + "], x = " + x[k]);
            }
        }
    }

    private static INDArray apply(boolean complementary, INDArray x) {
        INDArray z = Nd4j.valueArrayOf(x.shape(), SENTINEL, x.dataType());
        Nd4j.exec(complementary ? new Erfc(x, z) : new Erf(x, z));
        return z;
    }

    private static void assertRelative(String name, DataType type, double[] x, double[] expected, double tolerance,
                                       INDArray actual) {
        assertEquals(type, actual.dataType(), name + " output type");
        double[] values = actual.dup().data().asDouble();
        assertEquals(expected.length, values.length, name + " length");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], values[i], tolerance * Math.abs(expected[i]),
                    type + " " + name + "(" + x[i] + ")");
        }
    }

    private static void assertSpecial(String name, DataType type, double[] x, double[] expected, INDArray actual) {
        assertEquals(type, actual.dataType(), name + " output type");
        double[] values = actual.dup().data().asDouble();
        assertEquals(expected.length, values.length, name + " length");
        for (int i = 0; i < expected.length; i++) {
            String label = type + " " + name + "(" + x[i] + ") = " + values[i];
            if (Double.isNaN(expected[i])) {
                assertTrue(Double.isNaN(values[i]), label);
            } else {
                assertEquals(expected[i], values[i], 0.0, label);
            }
        }
    }

    private static double[] widen(float[] values) {
        double[] widened = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            widened[i] = values[i];
        }
        return widened;
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
