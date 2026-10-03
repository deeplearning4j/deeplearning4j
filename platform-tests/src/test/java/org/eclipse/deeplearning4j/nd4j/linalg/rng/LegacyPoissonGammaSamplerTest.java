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
package org.eclipse.deeplearning4j.nd4j.linalg.rng;

import org.eclipse.deeplearning4j.nd4j.linalg.api.ops.random.impl.GammaDistribution;
import org.eclipse.deeplearning4j.nd4j.linalg.api.ops.random.impl.PoissonDistribution;
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
import org.nd4j.linalg.api.ops.RandomOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The legacy random ops 15 (PoissonDistribution) and 16 (GammaDistribution) sample their distributions: they used to
 * evaluate a distribution function at a uniform draw over (-max / 10, max / 10), which is 0 or about 1 almost
 * everywhere. Element e draws from its own stream of the generator, draw j at index e * 2^16 + j, with the samplers the
 * random_poisson and random_gamma custom ops use.
 *
 * Below lambda 10 a Poisson sample multiplies uniform draws until the product falls to exp(-lambda), which this test
 * reproduces exactly from the Philox reference. The other samplers take logarithms, roots and powers that backends
 * round differently, so they are checked by their moments over many samples (bounds of six standard errors) and, for
 * Poisson, by a chi-square test against the probability mass function. Every check runs at fixed generator states, so
 * its outcome is the same on every run.
 */
@Tag(TagNames.RNG)
@NativeTag
public class LegacyPoissonGammaSamplerTest extends BaseNd4jTestWithBackends {

    private static final long ROOT = 0x0123456789abcdefL;
    private static final long NODE = 0x76543210fedcba98L;
    private static final int LENGTH = 4099;
    private static final int SAMPLES = 1 << 17;
    /** Draw j of element e is the generator's value at e * DRAWS_PER_ELEMENT + j. */
    private static final long DRAWS_PER_ELEMENT = 1L << 16;

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void poissonBelowTenMultipliesDraws(Nd4jBackend backend) {
        DataType[] types = sixteenBitStorage()
                ? new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.HALF, DataType.BFLOAT16}
                : new DataType[]{DataType.FLOAT, DataType.DOUBLE};
        for (DataType type : types) {
            for (double lambda : new double[]{0.5, 3.0, 9.5}) {
                double[] expected = new double[LENGTH];
                for (int e = 0; e < LENGTH; e++) expected[e] = knuth(type, lambda, e);
                INDArray actual = run(new PoissonDistribution(Nd4j.create(type, LENGTH), lambda));
                assertExact("Poisson(" + lambda + ") " + type, expected, actual);
            }
        }
    }

    /** One rate per element: each element samples with its own rate, from its own stream. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void poissonWithPerElementRates(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            double[] lambdas = new double[LENGTH];
            double[] expected = new double[LENGTH];
            for (int e = 0; e < LENGTH; e++) {
                lambdas[e] = 0.25 + (e % 37) / 4.0;
                expected[e] = knuth(type, lambdas[e], e);
            }
            // DOUBLE rates for a FLOAT output are cast to the output's type
            INDArray rates = Nd4j.createFromArray(lambdas);
            INDArray actual = run(new PoissonDistribution(Nd4j.create(type, LENGTH), rates));
            assertExact("Poisson with per-element rates " + type, expected, actual);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void poissonMomentsAndMass(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (double lambda : new double[]{0.5, 3.0, 9.5, 10.0, 25.0, 250.0, 5000.0}) {
                double[] values = sample(new PoissonDistribution(Nd4j.create(type, SAMPLES), lambda), 11);
                String label = "Poisson(" + lambda + ") " + type;
                for (double v : values) {
                    assertTrue(v >= 0 && v == Math.rint(v), label + ": not a count: " + v);
                }
                assertMoments(label, values, lambda, lambda, lambda + 3 * lambda * lambda);
                if (lambda <= 250) assertPoissonMass(label, values, lambda);
            }
        }
        if (!sixteenBitStorage()) return;
        // 16-bit outputs hold the counts exactly while they stay below 2048 (HALF) and 256 (BFLOAT16)
        assertMoments("Poisson(25) HALF", sample(new PoissonDistribution(Nd4j.create(DataType.HALF, SAMPLES), 25.0), 12),
                25.0, 25.0, 25.0 + 3 * 25.0 * 25.0);
        assertMoments("Poisson(25) BFLOAT16",
                sample(new PoissonDistribution(Nd4j.create(DataType.BFLOAT16, SAMPLES), 25.0), 13),
                25.0, 25.0, 25.0 + 3 * 25.0 * 25.0);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void gammaMoments(Nd4jBackend backend) {
        double[][] parameters = {{0.25, 1.0}, {0.9, 2.0}, {1.0, 0.5}, {2.5, 3.0}, {30.0, 1.5}};
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (double[] p : parameters) {
                double alpha = p[0], beta = p[1];
                double[] values = sample(new GammaDistribution(Nd4j.create(type, SAMPLES), alpha, beta), 21);
                String label = "Gamma(" + alpha + ", " + beta + ") " + type;
                for (double v : values) assertTrue(v > 0 && !Double.isInfinite(v), label + ": " + v);
                assertGammaMoments(label, values, alpha, beta);
            }
        }
        if (!sixteenBitStorage()) return;
        double[] half = sample(new GammaDistribution(Nd4j.create(DataType.HALF, SAMPLES), 2.5, 3.0), 22);
        assertGammaMoments("Gamma(2.5, 3) HALF", half, 2.5, 3.0);
    }

    /** One shape per element (x), and one shape and one rate per element (x and y). */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void gammaWithPerElementParameters(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            double[] alphas = new double[SAMPLES];
            double[] betas = new double[SAMPLES];
            for (int e = 0; e < SAMPLES; e++) {
                alphas[e] = e % 2 == 0 ? 0.5 : 4.0;
                betas[e] = e % 2 == 0 ? 2.0 : 0.25;
            }
            double[] shapes = sample(new GammaDistribution(Nd4j.create(type, SAMPLES),
                    Nd4j.createFromArray(alphas).castTo(type), 2.0), 31);
            assertGammaMoments("Gamma(0.5, 2) from x " + type, every(shapes, 0), 0.5, 2.0);
            assertGammaMoments("Gamma(4, 2) from x " + type, every(shapes, 1), 4.0, 2.0);

            double[] both = sample(new GammaDistribution(Nd4j.create(type, SAMPLES), Nd4j.createFromArray(alphas),
                    Nd4j.createFromArray(betas)), 32);
            assertGammaMoments("Gamma(0.5, 2) from x and y " + type, every(both, 0), 0.5, 2.0);
            assertGammaMoments("Gamma(4, 0.25) from x and y " + type, every(both, 1), 4.0, 0.25);
        }
    }

    /**
     * Element i of the output draws at logical index i whatever the arrays' order: the CPU path for three arrays of one
     * layout drew at memory offset i, which for 'f' arrays is another element.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void fOrderOperandsDrawLikeCOrder(Nd4jBackend backend) {
        double[][] alphas = new double[6][7];
        double[][] betas = new double[6][7];
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 7; j++) {
                alphas[i][j] = 0.5 + ((i * 7 + j) % 5) * 0.75;
                betas[i][j] = 0.25 + ((i * 3 + j) % 4) * 0.5;
            }
        }
        INDArray alphaC = Nd4j.createFromArray(alphas).castTo(DataType.FLOAT);
        INDArray betaC = Nd4j.createFromArray(betas).castTo(DataType.FLOAT);
        INDArray zC = run(new GammaDistribution(Nd4j.create(DataType.FLOAT, new long[]{6, 7}, 'c'), alphaC, betaC));
        INDArray zF = run(new GammaDistribution(Nd4j.create(DataType.FLOAT, new long[]{6, 7}, 'f'), alphaC.dup('f'),
                betaC.dup('f')));
        assertEquals(zC, zF);
    }

    /** A rate of 0 gives 0; a negative rate, shape or Gamma rate has no distribution: NaN. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void degenerateParameters(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (double v : run(new PoissonDistribution(Nd4j.create(type, 64), 0.0)).data().asDouble()) {
                assertEquals(0.0, v, "Poisson(0) " + type);
            }
            for (double v : run(new PoissonDistribution(Nd4j.create(type, 64), -1.0)).data().asDouble()) {
                assertTrue(Double.isNaN(v), "Poisson(-1) " + type + ": " + v);
            }
            for (double[] p : new double[][]{{0.0, 1.0}, {-2.0, 1.0}, {2.0, 0.0}, {2.0, -1.0}}) {
                for (double v : run(new GammaDistribution(Nd4j.create(type, 64), p[0], p[1])).data().asDouble()) {
                    assertTrue(Double.isNaN(v), "Gamma(" + p[0] + ", " + p[1] + ") " + type + ": " + v);
                }
            }
        }
    }

    /** The same generator state gives the same samples; the generator moves on after each fill. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void reproducibleAndAdvancing(Nd4jBackend backend) {
        INDArray first = run(new GammaDistribution(Nd4j.create(DataType.FLOAT, 1000), 1.5, 2.0));
        INDArray again = run(new GammaDistribution(Nd4j.create(DataType.FLOAT, 1000), 1.5, 2.0));
        assertEquals(first, again);
        INDArray next = Nd4j.getExecutioner().exec(new GammaDistribution(Nd4j.create(DataType.FLOAT, 1000), 1.5, 2.0),
                Nd4j.getRandom());
        assertNotEquals(first, next);

        INDArray poisson = run(new PoissonDistribution(Nd4j.create(DataType.FLOAT, 1000), 40.0));
        INDArray poissonNext = Nd4j.getExecutioner().exec(
                new PoissonDistribution(Nd4j.create(DataType.FLOAT, 1000), 40.0), Nd4j.getRandom());
        assertNotEquals(poisson, poissonNext);
    }

    /** In a SameDiff graph the ops run through the native legacy random op wrapper. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void inSameDiff(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable poisson = new PoissonDistribution(sd, 4.0, DataType.FLOAT, new long[]{SAMPLES}).outputVariable();
        SDVariable gamma = new GammaDistribution(sd, 2.0, 0.5, DataType.FLOAT, new long[]{SAMPLES}).outputVariable();
        Nd4j.getRandom().setSeed(41);
        INDArray poissonValues = poisson.eval();
        assertMoments("Poisson(4) in SameDiff", poissonValues.data().asDouble(), 4.0, 4.0, 4.0 + 3 * 16.0);
        INDArray gammaValues = gamma.eval();
        assertGammaMoments("Gamma(2, 0.5) in SameDiff", gammaValues.data().asDouble(), 2.0, 0.5);
    }

    /**
     * A random op reads its x and y arrays as its output's type: an array of another type is rejected instead of read
     * as garbage. The Poisson and Gamma ops cast their parameter arrays themselves.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void parameterArraysOfAnotherTypeAreRejected(Nd4jBackend backend) {
        INDArray z = Nd4j.create(DataType.FLOAT, 16);
        INDArray probabilities = Nd4j.valueArrayOf(new long[]{16}, 0.5, DataType.DOUBLE);
        assertThrows(RuntimeException.class,
                () -> Nd4j.getExecutioner().exec(new BernoulliDistribution(z, probabilities), Nd4j.getRandom()));
    }

    /**
     * Whether the backend stores HALF and BFLOAT16 arrays: Vulkan has no BFLOAT16, and HALF only on devices with 16-bit
     * storage, so the 16-bit cases run on the other backends.
     */
    private static boolean sixteenBitStorage() {
        return Nd4j.getExecutioner().type() != OpExecutioner.ExecutionerType.VULKAN;
    }

    /** The Knuth sample of element e at the generator states ROOT and NODE, computed in the op's type. */
    private static double knuth(DataType type, double lambda, long element) {
        boolean doubles = type == DataType.DOUBLE;
        double lam = typed(type, lambda);
        double bound = doubles ? Math.exp(-lam) : (float) Math.exp(-(float) lam);
        long draw = 0;
        double product = streamUniform(doubles, element, draw++);
        int count = 0;
        while (product > bound) {
            count++;
            double u = streamUniform(doubles, element, draw++);
            // the product of two floats is exact in double; rounding it to float is the float product
            product = doubles ? product * u : (float) (product * u);
        }
        return count;
    }

    /** helpers::streamUniform: 1 - relativeT(e * 2^16 + j), in (0, 1], in float for every type but DOUBLE. */
    private static double streamUniform(boolean doubles, long element, long draw) {
        long index = element * DRAWS_PER_ELEMENT + (draw & (DRAWS_PER_ELEMENT - 1));
        return doubles ? 1.0 - PhiloxReference.uniformDouble(ROOT, NODE, index)
                : 1.0f - PhiloxReference.uniformFloat(ROOT, NODE, index);
    }

    /** A parameter as the op receives it: in the output's type. */
    private static double typed(DataType type, double value) {
        switch (type) {
            case DOUBLE:
                return value;
            case FLOAT:
                return (float) value;
            default:
                return Nd4j.scalar(DataType.DOUBLE, value).castTo(type).getDouble(0);
        }
    }

    private static INDArray run(RandomOp op) {
        Nd4j.getRandom().setStates(ROOT, NODE);
        return Nd4j.getExecutioner().exec(op, Nd4j.getRandom());
    }

    private static double[] sample(RandomOp op, long seed) {
        Nd4j.getRandom().setSeed(seed);
        return Nd4j.getExecutioner().exec(op, Nd4j.getRandom()).data().asDouble();
    }

    /** Every second value, from offset 0 or 1. */
    private static double[] every(double[] values, int offset) {
        double[] out = new double[values.length / 2];
        for (int i = 0; i < out.length; i++) out[i] = values[2 * i + offset];
        return out;
    }

    private static void assertExact(String label, double[] expected, INDArray actual) {
        double[] values = actual.data().asDouble();
        assertEquals(expected.length, values.length, label + " length");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], values[i], 0.0, label + " element " + i);
        }
    }

    /**
     * The sample mean and variance against the distribution's, within six standard errors: the mean's is
     * sqrt(variance / n), the sample variance's sqrt((mu4 - variance^2) / n) with mu4 the fourth central moment.
     */
    private static void assertMoments(String label, double[] values, double mean, double variance,
                                      double fourthCentralMoment) {
        int n = values.length;
        double sum = 0;
        for (double v : values) sum += v;
        double sampleMean = sum / n;
        double squares = 0;
        for (double v : values) squares += (v - sampleMean) * (v - sampleMean);
        double sampleVariance = squares / (n - 1);
        double meanError = Math.sqrt(variance / n);
        double varianceError = Math.sqrt((fourthCentralMoment - variance * variance) / n);
        assertEquals(mean, sampleMean, 6 * meanError, label + " mean");
        assertEquals(variance, sampleVariance, 6 * varianceError, label + " variance");
    }

    /** Gamma(alpha, beta): mean alpha / beta, variance alpha / beta^2, fourth central moment 3 alpha (alpha + 2) / beta^4. */
    private static void assertGammaMoments(String label, double[] values, double alpha, double beta) {
        double b2 = beta * beta;
        assertMoments(label, values, alpha / beta, alpha / b2, 3 * alpha * (alpha + 2) / (b2 * b2));
    }

    /**
     * Pearson's chi-square of the sample counts against Poisson(lambda)'s probability mass, over the counts whose
     * expected frequency is at least 20 (the rest merged into the two tails). The bound, the degrees of freedom plus
     * eight of their standard deviations, holds for a correct sampler with probability far above 1 - 1e-9.
     */
    private static void assertPoissonMass(String label, double[] values, double lambda) {
        int n = values.length;
        int max = (int) (lambda + 20 * Math.sqrt(lambda) + 20);
        double[] mass = new double[max + 1];
        double logFactorial = 0;
        for (int k = 0; k <= max; k++) {
            if (k > 0) logFactorial += Math.log(k);
            mass[k] = Math.exp(-lambda + k * Math.log(lambda) - logFactorial);
        }
        long[] counts = new long[max + 1];
        for (double v : values) counts[(int) Math.min(v, max)]++;
        // bins [low, high] each have expected count >= 20; below low and above high merge into the end bins
        int low = 0;
        double lowTail = mass[0];
        while (lowTail * n < 20) lowTail += mass[++low];
        int high = max;
        double highTail = 1.0;
        for (int k = 0; k < high; k++) highTail -= mass[k];
        while (highTail * n < 20) highTail += mass[--high];
        double chi2 = 0;
        int bins = 0;
        long observedLow = 0;
        for (int k = 0; k <= low; k++) observedLow += counts[k];
        chi2 += square(observedLow - lowTail * n) / (lowTail * n);
        bins++;
        for (int k = low + 1; k < high; k++) {
            double expected = mass[k] * n;
            chi2 += square(counts[k] - expected) / expected;
            bins++;
        }
        long observedHigh = 0;
        for (int k = high; k <= max; k++) observedHigh += counts[k];
        chi2 += square(observedHigh - highTail * n) / (highTail * n);
        bins++;
        int degrees = bins - 1;
        assertTrue(chi2 < degrees + 8 * Math.sqrt(2.0 * degrees),
                label + ": chi-square " + chi2 + " over " + degrees + " degrees of freedom");
    }

    private static double square(double value) {
        return value * value;
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
