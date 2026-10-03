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

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * betainc, the regularized incomplete beta function I_x(a, b), against a double-precision continued fraction, for
 * more elements than a CUDA block has threads and for strided views. The CUDA kernel computes an element per block;
 * its launch gave the element count as the threads per block instead (an invalid launch beyond 1024 elements, and
 * blocks past the element count reading and writing beyond the arrays).
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class BetaIncTest extends BaseNd4jTestWithBackends {

    /** ln Gamma(z) for z > 0 (Lanczos, g = 7, n = 9). */
    private static double logGamma(double z) {
        double[] c = {0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
                -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
                1.5056327351493116e-7};
        if (z < 0.5)
            return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
        z -= 1;
        double x = c[0];
        for (int i = 1; i < 9; i++)
            x += c[i] / (z + i);
        double t = z + 7.5;
        return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
    }

    /** The continued fraction of I_x(a, b) (modified Lentz). */
    private static double betaContinuedFraction(double a, double b, double x) {
        double tiny = 1e-300;
        double c = 1, d = 1 - (a + b) * x / (a + 1);
        if (Math.abs(d) < tiny)
            d = tiny;
        d = 1 / d;
        double h = d;
        for (int m = 1; m <= 10000; m++) {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2));
            d = 1 + aa * d;
            if (Math.abs(d) < tiny)
                d = tiny;
            c = 1 + aa / c;
            if (Math.abs(c) < tiny)
                c = tiny;
            d = 1 / d;
            h *= d * c;
            aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1));
            d = 1 + aa * d;
            if (Math.abs(d) < tiny)
                d = tiny;
            c = 1 + aa / c;
            if (Math.abs(c) < tiny)
                c = tiny;
            d = 1 / d;
            double del = d * c;
            h *= del;
            if (Math.abs(del - 1) < 1e-16)
                break;
        }
        return h;
    }

    private static double betaincReference(double a, double b, double x) {
        if (x <= 0)
            return 0;
        if (x >= 1)
            return 1;
        double front = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
        if (x < (a + 1) / (a + b + 2))
            return front * betaContinuedFraction(a, b, x) / a;
        return 1 - front * betaContinuedFraction(b, a, 1 - x) / b;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void betaincMatchesTheContinuedFraction(Nd4jBackend backend) {
        Random rng = new Random(31);
        int rows = 3, cols = 700;  // 2100 elements: more than a CUDA block's threads
        double[][] a = new double[rows][cols], b = new double[rows][cols], x = new double[rows][cols];
        double[][] expected = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                a[i][j] = 0.5 + 4.5 * rng.nextDouble();
                b[i][j] = 0.5 + 4.5 * rng.nextDouble();
                x[i][j] = rng.nextDouble();
                expected[i][j] = betaincReference(a[i][j], b[i][j], x[i][j]);
            }
        INDArray expectedArr = Nd4j.createFromArray(expected);
        INDArray aArr = Nd4j.createFromArray(a), bArr = Nd4j.createFromArray(b), xArr = Nd4j.createFromArray(x);

        INDArray[] dense = Nd4j.exec(DynamicCustomOp.builder("betainc").addInputs(aArr, bArr, xArr).build());
        double maxDiff = dense[0].sub(expectedArr).amaxNumber().doubleValue();
        assertTrue(maxDiff <= 1e-10, "betainc of dense operands: max |diff| " + maxDiff);

        // the same operands as stepped views of arrays twice as wide
        INDArray[] stepped = new INDArray[3];
        INDArray[] dense3 = {aArr, bArr, xArr};
        for (int k = 0; k < 3; k++) {
            INDArray parent = Nd4j.valueArrayOf(new long[]{rows, 2 * cols}, Double.NaN, DataType.DOUBLE);
            stepped[k] = parent.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 2 * cols)).assign(dense3[k]);
        }
        INDArray[] fromViews = Nd4j.exec(DynamicCustomOp.builder("betainc").addInputs(stepped).build());
        maxDiff = fromViews[0].sub(expectedArr).amaxNumber().doubleValue();
        assertTrue(maxDiff <= 1e-10, "betainc of stepped views: max |diff| " + maxDiff);
    }
}
