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
import org.nd4j.linalg.api.ops.custom.Lgamma;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * lgamma is log |Gamma(x)|. Below 1 the native Gamma is Gamma(1 + x) / x with x kept exact; below 0.001 it was
 * once approximated as 1 / (x (1 + 0.5772 x)), up to 6.6e-7 off. Below 0 it goes through the reflection formula;
 * it was NaN there.
 */
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@NativeTag
public class LogGammaTest extends BaseNd4jTestWithBackends {

    private static final double[] POINTS = {1e-300, 1e-7, 1e-4, 5e-4, 9e-4, 0.25, 0.5, 1, 1.5, 2, 3.7, 11.9, 12, 50,
            1000, -0.5, -1.5, -2.25, -7.75, -100.5};

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void matchesCommonsMathInDouble(Nd4jBackend backend) {
        INDArray out = Nd4j.exec(new Lgamma(Nd4j.createFromArray(POINTS)))[0];
        assertEquals(DataType.DOUBLE, out.dataType());
        for (int i = 0; i < POINTS.length; i++) {
            double expected = logAbsGamma(POINTS[i]);
            assertEquals(expected, out.getDouble(i), 1e-13 * Math.abs(expected) + 1e-14, "lgamma(" + POINTS[i] + ")");
        }
    }

    /**
     * Below 0 near 0 and near the integers, where the reflection's sin(pi x) needs x's fraction: sin(pi x) once reduced x
     * as x - 2 floor(x / 2), which turned -1e-300 into 2.0 and lost it (lgamma(-1e-300) was 37.09, not 690.78), and lost
     * digits next to every negative integer. The references are CPython's math.lgamma, which reduces exactly.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void negativeArgumentsNearIntegers(Nd4jBackend backend) {
        double[] x = {-1e-300, -1e-10, -2.9999999999, -3.0000000001, -2.5, -7.25, -100.000001};
        double[] expected = {690.7755278982137, 23.025850929998178, 21.23409137809765, 21.234091377846426,
                -0.05624371649767457, -7.541883443475751, -349.92386960523464};
        INDArray out = Nd4j.exec(new Lgamma(Nd4j.createFromArray(x)))[0];
        for (int i = 0; i < x.length; i++) {
            assertEquals(expected[i], out.getDouble(i), 1e-12 * Math.abs(expected[i]) + 1e-14, "lgamma(" + x[i] + ")");
        }
    }

    /** log |Gamma(x)|: Commons Math above 0, and log(pi / |sin(pi x)|) - log Gamma(1 - x) below. */
    private static double logAbsGamma(double x) {
        if (x > 0) {
            return Gamma.logGamma(x);
        }
        double reduced = x - 2 * Math.floor(x / 2);
        return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * reduced))) - Gamma.logGamma(1 - x);
    }
}
