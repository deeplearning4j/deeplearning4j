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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.RandomOp;
import org.nd4j.linalg.api.ops.random.impl.AlphaDropOut;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.api.ops.random.impl.BinomialDistribution;
import org.nd4j.linalg.api.ops.random.impl.BinomialDistributionEx;
import org.nd4j.linalg.api.ops.random.impl.Choice;
import org.nd4j.linalg.api.ops.random.impl.DropOut;
import org.nd4j.linalg.api.ops.random.impl.DropOutInverted;
import org.nd4j.linalg.api.ops.random.impl.GaussianDistribution;
import org.nd4j.linalg.api.ops.random.impl.Linspace;
import org.nd4j.linalg.api.ops.random.impl.LogNormalDistribution;
import org.nd4j.linalg.api.ops.random.impl.ProbablisticMerge;
import org.nd4j.linalg.api.ops.random.impl.TruncatedNormalDistribution;
import org.nd4j.linalg.api.ops.random.impl.UniformDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The legacy random ops (RANDOM_OPS in libnd4j's legacy_ops.h) compute each element from the
 * generator's Philox draws at fixed indices, so {@link PhiloxReference} reproduces them in Java.
 * Each test runs an op at known generator states and compares every element with the Java form of
 * the native algorithm (random_ops.h, special_random_ops.h): exactly where the op only compares,
 * selects and divides, and within an error bound where it computes, since a backend may fuse a
 * multiply-add and round log, sqrt, sin, cos and exp differently. The bound for a normal sample
 * propagates an ulp of each draw through the Box-Muller transform.
 */
@Tag(TagNames.RNG)
@NativeTag
public class LegacyRandomOpsReferenceTest extends BaseNd4jTestWithBackends {

    private static final long ROOT = 0x0123456789abcdefL;
    private static final long NODE = 0x76543210fedcba98L;
    /** Odd, so one element of the Box-Muller pairs has no partner. */
    private static final int LENGTH = 4099;
    private static final DataType[] TYPES = {DataType.FLOAT, DataType.DOUBLE};
    /** The lower bound of the first Box-Muller draw: relativeT(index, 1e-5, 1). */
    private static final double EPSILON = 1e-5;
    /** Attempts the truncated normal sampler makes per element before it clamps. */
    private static final int TRUNCATED_ATTEMPTS = 64;

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void uniformIsTheDrawScaledToTheRange(Nd4jBackend backend) {
        double from = -2.5;
        double to = 4.0;
        for (DataType type : TYPES) {
            double[] expected = new double[LENGTH];
            double[] tolerance = new double[LENGTH];
            for (int i = 0; i < LENGTH; i++) {
                double scaled = multiply(type, draw(type, i), subtract(type, typed(type, to), typed(type, from)));
                expected[i] = add(type, typed(type, from), scaled);
                // A fused multiply-add (CUDA contracts from + draw * range into one) rounds once where this reference
                // rounds twice: they differ by up to half an ulp of the product, which near the range's zero crossing
                // is many ulps of the result.
                tolerance[i] = ulp(type, scaled) + 2 * ulp(type, expected[i]);
            }
            assertWithin("UniformDistribution", type, expected, tolerance,
                    run(new UniformDistribution(Nd4j.create(type, LENGTH), from, to)));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void dropOutZeroesWhereTheDrawReachesP(Nd4jBackend backend) {
        double p = 0.3;
        for (DataType type : TYPES) {
            INDArray x = input(type, 0);
            double[] xs = x.data().asDouble();
            double[] dropOut = new double[LENGTH];
            double[] inverted = new double[LENGTH];
            for (int i = 0; i < LENGTH; i++) {
                boolean drop = draw(type, i) >= typed(type, p);
                dropOut[i] = drop ? 0.0 : xs[i];
                inverted[i] = drop ? 0.0 : typed(type, xs[i] / typed(type, p));
            }
            assertExact("DropOut", type, dropOut, run(new DropOut(x, Nd4j.create(type, LENGTH), p)));
            assertExact("DropOutInverted", type, inverted,
                    run(new DropOutInverted(x, Nd4j.create(type, LENGTH), p)));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void alphaDropOutScalesKeptValuesAndReplacesDroppedOnes(Nd4jBackend backend) {
        double p = 0.4;
        double alpha = 1.6732632423543772;
        double alphaPrime = -1.7580993408473766;
        double beta = 0.25;
        for (DataType type : TYPES) {
            INDArray x = input(type, 0);
            double[] xs = x.data().asDouble();
            double a = typed(type, alpha);
            double b = typed(type, beta);
            double[] expected = new double[LENGTH];
            double[] tolerance = new double[LENGTH];
            for (int i = 0; i < LENGTH; i++) {
                double kept = draw(type, i) >= typed(type, p) ? typed(type, alphaPrime) : xs[i];
                expected[i] = add(type, multiply(type, a, kept), b);
                tolerance[i] = 2 * ulp(type, expected[i]) + ulp(type, multiply(type, a, kept));
            }
            assertWithin("AlphaDropOut", type, expected, tolerance,
                    run(new AlphaDropOut(x, Nd4j.create(type, LENGTH), p, alpha, alphaPrime, beta)));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void probabilisticMergeTakesYWhereTheDrawIsAtMostP(Nd4jBackend backend) {
        double p = 0.35;
        for (DataType type : TYPES) {
            INDArray x = input(type, 0);
            INDArray y = input(type, 1);
            double[] xs = x.data().asDouble();
            double[] ys = y.data().asDouble();
            double[] expected = new double[LENGTH];
            for (int i = 0; i < LENGTH; i++) {
                expected[i] = draw(type, i) <= typed(type, p) ? ys[i] : xs[i];
            }
            assertExact("ProbablisticMerge", type, expected,
                    run(new ProbablisticMerge(x, y, Nd4j.create(type, LENGTH), p)));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void choiceTakesTheFirstSourceWhoseCumulativeProbabilityReachesTheDraw(Nd4jBackend backend) {
        double[] source = {10, 20, 30, 40, 50};
        double[] probabilities = {0.1, 0.2, 0.3, 0.15, 0.25};
        for (DataType type : TYPES) {
            double[] expected = new double[LENGTH];
            for (int e = 0; e < LENGTH; e++) {
                double u = draw(type, e);
                double cumulative = 0.0;
                for (int f = 0; f < source.length; f++) {
                    cumulative = add(type, cumulative, typed(type, probabilities[f]));
                    if (u <= cumulative || f == source.length - 1) {
                        expected[e] = source[f];
                        break;
                    }
                }
            }
            assertExact("Choice", type, expected, run(new Choice(Nd4j.createFromArray(source).castTo(type),
                    Nd4j.createFromArray(probabilities).castTo(type), Nd4j.create(type, LENGTH))));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void bernoulliIsOneWhereTheProbabilityReachesTheDraw(Nd4jBackend backend) {
        double p = 0.37;
        for (DataType type : TYPES) {
            double[] probabilities = new double[LENGTH];
            for (int i = 0; i < LENGTH; i++) {
                probabilities[i] = typed(type, (i % 10) / 10.0 + 0.05);
            }
            double[] scalar = new double[LENGTH];
            double[] perElement = new double[LENGTH];
            for (int i = 0; i < LENGTH; i++) {
                scalar[i] = typed(type, p) >= draw(type, i) ? 1.0 : 0.0;
                perElement[i] = probabilities[i] >= draw(type, i) ? 1.0 : 0.0;
            }
            assertExact("BernoulliDistribution", type, scalar,
                    run(new BernoulliDistribution(Nd4j.create(type, LENGTH), p)));
            assertExact("BernoulliDistribution(probabilities)", type, perElement,
                    run(new BernoulliDistribution(Nd4j.create(type, LENGTH),
                            Nd4j.createFromArray(probabilities).castTo(type))));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void binomialCountsTheTrialsWhoseDrawIsBelowTheirProbability(Nd4jBackend backend) {
        int trials = 7;
        double p = 0.45;
        double[] trialProbabilities = {0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95};
        for (DataType type : TYPES) {
            double[] elementProbabilities = new double[LENGTH];
            for (int e = 0; e < LENGTH; e++) {
                elementProbabilities[e] = typed(type, ((e * 37) % 100) / 100.0);
            }
            double[] scalar = new double[LENGTH];
            double[] perTrial = new double[LENGTH];
            double[] perElement = new double[LENGTH];
            for (int e = 0; e < LENGTH; e++) {
                for (int t = 1; t <= trials; t++) {
                    double u = draw(type, (long) e * trials + t - 1);
                    if (u < typed(type, p)) scalar[e]++;
                    if (u < typed(type, trialProbabilities[t - 1])) perTrial[e]++;
                    if (u < elementProbabilities[e]) perElement[e]++;
                }
            }
            assertExact("BinomialDistribution", type, scalar,
                    run(new BinomialDistribution(Nd4j.create(type, LENGTH), trials, p)));
            assertExact("BinomialDistribution(probabilities)", type, perTrial,
                    run(new BinomialDistribution(Nd4j.create(type, LENGTH), trials,
                            Nd4j.createFromArray(trialProbabilities).castTo(type))));
            assertExact("BinomialDistributionEx", type, scalar,
                    run(new BinomialDistributionEx(Nd4j.create(type, LENGTH), trials, p)));
            assertExact("BinomialDistributionEx(probabilities)", type, perElement,
                    run(new BinomialDistributionEx(Nd4j.create(type, LENGTH), trials,
                            Nd4j.createFromArray(elementProbabilities).castTo(type))));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void linspaceInterpolatesTheRange(Nd4jBackend backend) {
        double from = -3.0;
        double to = 5.5;
        double step = 0.125;
        for (DataType type : TYPES) {
            double[] interpolated = new double[LENGTH];
            double[] stepped = new double[LENGTH];
            double[] interpolatedTolerance = new double[LENGTH];
            double[] steppedTolerance = new double[LENGTH];
            double first = typed(type, from);
            double last = typed(type, to);
            for (int i = 0; i < LENGTH; i++) {
                double fraction = divide(type, typed(type, i), subtract(type, LENGTH, 1.0));
                double head = multiply(type, first, subtract(type, 1.0, fraction));
                double tail = multiply(type, fraction, last);
                interpolated[i] = add(type, head, tail);
                interpolatedTolerance[i] = 2 * ulp(type, interpolated[i]) + ulp(type, head) + ulp(type, tail);
                double offset = multiply(type, typed(type, i), typed(type, step));
                stepped[i] = add(type, first, offset);
                steppedTolerance[i] = 2 * ulp(type, stepped[i]) + ulp(type, offset);
            }
            assertWithin("Linspace", type, interpolated, interpolatedTolerance,
                    run(new Linspace(Nd4j.create(type, LENGTH), from, to)));
            assertWithin("Linspace(step)", type, stepped, steppedTolerance,
                    run(new Linspace(Nd4j.create(type, LENGTH), from, to, step)));
        }
    }

    /**
     * Element e < middle is the cosine of the pair (e, e + middle) and element e + middle its sine,
     * where middle is half the length rounded up.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void gaussianIsTheBoxMullerPairOfTwoDraws(Nd4jBackend backend) {
        double mean = 1.5;
        double stddev = 0.75;
        for (DataType type : TYPES) {
            double[] means = means(type);
            double[][] scalar = boxMuller(type, mean, null, stddev, false);
            double[][] perElement = boxMuller(type, mean, means, stddev, false);
            assertWithin("GaussianDistribution", type, scalar[0], scalar[1],
                    run(new GaussianDistribution(Nd4j.create(type, LENGTH), mean, stddev)));
            assertWithin("GaussianDistribution(means)", type, perElement[0], perElement[1],
                    run(new GaussianDistribution(Nd4j.create(type, LENGTH),
                            Nd4j.createFromArray(means).castTo(type), stddev)));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void logNormalIsTheExponentialOfTheGaussian(Nd4jBackend backend) {
        double mean = -0.5;
        double stddev = 0.6;
        for (DataType type : TYPES) {
            double[] means = means(type);
            double[][] scalar = boxMuller(type, mean, null, stddev, true);
            double[][] perElement = boxMuller(type, mean, means, stddev, true);
            assertWithin("LogNormalDistribution", type, scalar[0], scalar[1],
                    run(new LogNormalDistribution(Nd4j.create(type, LENGTH), mean, stddev)));
            assertWithin("LogNormalDistribution(means)", type, perElement[0], perElement[1],
                    run(new LogNormalDistribution(Nd4j.create(type, LENGTH),
                            Nd4j.createFromArray(means).castTo(type), stddev)));
        }
    }

    /**
     * Attempt k for element e draws the pair (2 (k length + e), 2 (k length + e) + 1) and takes the
     * cosine sample if it lies within two standard deviations, so an element depends on no other.
     * Elements whose acceptance a rounding could flip are not compared.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void truncatedNormalRedrawsUntilWithinTwoStandardDeviations(Nd4jBackend backend) {
        double mean = 2.0;
        double stddev = 0.5;
        for (DataType type : TYPES) {
            double[] means = means(type);
            for (boolean perElementMean : new boolean[]{false, true}) {
                double[] expected = new double[LENGTH];
                double[] tolerance = new double[LENGTH];
                int ambiguous = 0;
                for (int e = 0; e < LENGTH; e++) {
                    double elementMean = perElementMean ? means[e] : typed(type, mean);
                    double sample = Double.NaN;
                    double sampleTolerance = 0.0;
                    for (int k = 0; k < TRUNCATED_ATTEMPTS; k++) {
                        long pair = 2L * ((long) k * LENGTH + e);
                        double r0 = boxMullerDraw(type, pair);
                        double r1 = draw(type, pair + 1);
                        double n = normal(r0, r1, false);
                        double nTolerance = normalTolerance(type, r0, r1, n);
                        if (Math.abs(Math.abs(n) - 2.0) <= nTolerance) {
                            sample = Double.NaN;
                            break;
                        }
                        if (Math.abs(n) <= 2.0) {
                            sample = n;
                            sampleTolerance = nTolerance;
                            break;
                        }
                    }
                    if (Double.isNaN(sample)) {
                        ambiguous++;
                        expected[e] = Double.NaN;
                        continue;
                    }
                    expected[e] = elementMean + typed(type, stddev) * sample;
                    tolerance[e] = Math.abs(typed(type, stddev)) * sampleTolerance + 4 * ulp(type, expected[e]);
                }
                assertTrue(ambiguous < 8, ambiguous + " ambiguous elements");
                String name = perElementMean ? "TruncatedNormalDistribution(means)" : "TruncatedNormalDistribution";
                INDArray z = Nd4j.create(type, LENGTH);
                INDArray out = run(perElementMean
                        ? new TruncatedNormalDistribution(z, Nd4j.createFromArray(means).castTo(type), stddev)
                        : new TruncatedNormalDistribution(z, mean, stddev));
                double[] values = out.data().asDouble();
                for (int e = 0; e < LENGTH; e++) {
                    double elementMean = perElementMean ? means[e] : typed(type, mean);
                    double halfWidth = 2 * Math.abs(typed(type, stddev));
                    double bound = halfWidth + 4 * ulp(type, Math.abs(elementMean) + halfWidth);
                    assertTrue(Math.abs(values[e] - elementMean) <= bound,
                            name + " " + type + " index " + e + ": " + values[e] + " is beyond two standard deviations of " + elementMean);
                }
                assertWithin(name, type, expected, tolerance, out);
            }
        }
    }

    /**
     * A HALF output draws relativeT&lt;float&gt;(index) rounded to half (RandomGenerator::relativeT&lt;float16&gt;) and
     * compares that: a draw within half an ulp below the threshold rounds onto it. DropOut, ProbablisticMerge,
     * BernoulliDistribution and BinomialDistribution decide by the rounded draw; with this many elements some decide
     * otherwise by the float draw, which the test checks so that it tells the two apart.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void halfOutputsCompareTheDrawRoundedToHalf(Nd4jBackend backend) {
        DataType type = DataType.HALF;
        int length = 1 << 16;
        int trials = 3;
        double p = 0.3;
        double threshold = Nd4j.scalar(DataType.FLOAT, (float) p).castTo(type).getDouble(0);
        double[] draws = halfDraws(length * trials);
        INDArray x = input(type, 0, length);
        INDArray y = input(type, 1, length);
        double[] xs = x.data().asDouble();
        double[] ys = y.data().asDouble();
        double[] dropOut = new double[length];
        double[] merge = new double[length];
        double[] bernoulli = new double[length];
        double[] binomial = new double[length];
        int flipped = 0;
        for (int i = 0; i < length; i++) {
            double u = draws[i];
            if ((u >= threshold) != (PhiloxReference.uniformFloat(ROOT, NODE, i) >= threshold)) {
                flipped++;
            }
            dropOut[i] = u >= threshold ? 0.0 : xs[i];
            merge[i] = u <= threshold ? ys[i] : xs[i];
            bernoulli[i] = threshold >= u ? 1.0 : 0.0;
            for (int t = 1; t <= trials; t++) {
                if (draws[i * trials + t - 1] < threshold) binomial[i]++;
            }
        }
        assertTrue(flipped > 0, "no draw rounds across the threshold: the rounded and the float draw decide alike");
        assertExact("DropOut", type, dropOut, run(new DropOut(x, Nd4j.create(type, length), p)));
        assertExact("ProbablisticMerge", type, merge, run(new ProbablisticMerge(x, y, Nd4j.create(type, length), p)));
        assertExact("BernoulliDistribution", type, bernoulli,
                run(new BernoulliDistribution(Nd4j.create(type, length), p)));
        assertExact("BinomialDistribution", type, binomial,
                run(new BinomialDistribution(Nd4j.create(type, length), trials, p)));
    }

    /**
     * A HALF log-normal saturates at 65504, the largest half, as sd_exp&lt;float16&gt; does, where converting the float
     * exponential gave inf. Every element is finite, and those whose exponent lies above log(2 * 65504) (the half
     * arithmetic moves an exponent by a few hundredths) read 65504.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void halfLogNormalSaturatesAtTheLargestHalf(Nd4jBackend backend) {
        double mean = 10.0;
        double stddev = 1.0;
        double largest = 65504.0;
        double[] values = run(new LogNormalDistribution(Nd4j.create(DataType.HALF, LENGTH), mean, stddev))
                .data().asDouble();
        int middle = LENGTH / 2 + LENGTH % 2;
        // the last pair's second draw lies past the end of an odd length, as in boxMuller
        double[] draws = halfDraws(2 * middle);
        double epsilon = Nd4j.scalar(DataType.FLOAT, (float) EPSILON).castTo(DataType.HALF).getDouble(0);
        int saturated = 0;
        for (int e = 0; e < LENGTH; e++) {
            assertTrue(values[e] > 0.0 && values[e] <= largest, "LogNormalDistribution HALF index " + e + ": " + values[e]);
            boolean sine = e >= middle;
            int first = sine ? e - middle : e;
            double r0 = epsilon + draws[first] * (1.0 - epsilon);
            double r1 = epsilon + draws[first + middle] * (1.0 - epsilon);
            if (mean + stddev * normal(r0, r1, sine) > Math.log(2 * largest)) {
                assertEquals(largest, values[e], 0.0, "LogNormalDistribution HALF index " + e);
                saturated++;
            }
        }
        assertTrue(saturated > 0, "no sample lies above log(2 * 65504)");
    }

    /** relativeT&lt;float16&gt;(index) for indices 0 .. count - 1: the float draws, rounded to half by the backend. */
    private static double[] halfDraws(int count) {
        float[] draws = new float[count];
        for (int i = 0; i < count; i++) {
            draws[i] = PhiloxReference.uniformFloat(ROOT, NODE, i);
        }
        return Nd4j.createFromArray(draws).castTo(DataType.HALF).castTo(DataType.DOUBLE).data().asDouble();
    }

    /**
     * Expected Box-Muller samples and their error bounds: [0] the values, [1] the bounds. Means,
     * when given, are per element; the exponential gives the log-normal.
     */
    private static double[][] boxMuller(DataType type, double mean, double[] means, double stddev, boolean exponential) {
        int middle = LENGTH / 2 + LENGTH % 2;
        double sigma = typed(type, stddev);
        double[] expected = new double[LENGTH];
        double[] tolerance = new double[LENGTH];
        for (int e = 0; e < LENGTH; e++) {
            boolean sine = e >= middle;
            int first = sine ? e - middle : e;
            double r0 = boxMullerDraw(type, first);
            double r1 = boxMullerDraw(type, first + middle);
            double n = normal(r0, r1, sine);
            double nTolerance = normalTolerance(type, r0, r1, n);
            double elementMean = means != null ? means[e] : typed(type, mean);
            double value = elementMean + sigma * n;
            double valueTolerance = Math.abs(sigma) * nTolerance + 4 * ulp(type, value);
            if (exponential) {
                expected[e] = Math.exp(value);
                tolerance[e] = expected[e] * (valueTolerance + 4 * ulp(type, 1.0)) + 4 * ulp(type, expected[e]);
            } else {
                expected[e] = value;
                tolerance[e] = valueTolerance;
            }
        }
        return new double[][]{expected, tolerance};
    }

    /** relativeT(index, 1e-5, 1) in the op's type: epsilon + u (1 - epsilon). */
    private static double boxMullerDraw(DataType type, long index) {
        double epsilon = typed(type, EPSILON);
        return add(type, epsilon, multiply(type, draw(type, index), subtract(type, 1.0, epsilon)));
    }

    private static double normal(double r0, double r1, boolean sine) {
        double radius = Math.sqrt(-2.0 * Math.log(r0));
        double angle = 2.0 * Math.PI * r1;
        return radius * (sine ? Math.sin(angle) : Math.cos(angle));
    }

    /**
     * How far a backend's sample may lie from the reference's: an ulp of each draw (a fused
     * multiply-add rounds once), propagated through sqrt(-2 log r0) and the angle 2 pi r1, plus the
     * rounding of log, sqrt, sin and cos.
     */
    private static double normalTolerance(DataType type, double r0, double r1, double n) {
        double radius = Math.sqrt(-2.0 * Math.log(r0));
        double radiusSlope = 1.0 / (r0 * radius);
        double angleError = 2.0 * Math.PI * 2 * ulp(type, r1) + 8 * ulp(type, 2.0 * Math.PI);
        return 2 * ulp(type, r0) * radiusSlope + radius * (angleError + 8 * ulp(type, 1.0)) + 8 * ulp(type, n);
    }

    /** Per-element means for the samplers' array form. */
    private static double[] means(DataType type) {
        double[] means = new double[LENGTH];
        for (int e = 0; e < LENGTH; e++) {
            means[e] = typed(type, ((e * 13) % 41) / 8.0 - 2.5);
        }
        return means;
    }

    /** Deterministic op input; variant 1 differs from variant 0 at every index. */
    private static INDArray input(DataType type, int variant) {
        return input(type, variant, LENGTH);
    }

    private static INDArray input(DataType type, int variant, int length) {
        double[] values = new double[length];
        for (int i = 0; i < length; i++) {
            values[i] = (i % 97) - 48 + 0.25 * (i % 7) + (variant == 0 ? 0.5 : -100.0);
        }
        return Nd4j.createFromArray(values).castTo(type);
    }

    private static INDArray run(RandomOp op) {
        Nd4j.getRandom().setStates(ROOT, NODE);
        return Nd4j.getExecutioner().exec(op, Nd4j.getRandom());
    }

    /** RandomGenerator::relativeT&lt;T&gt;(index), widened to double. */
    private static double draw(DataType type, long index) {
        return type == DataType.FLOAT ? PhiloxReference.uniformFloat(ROOT, NODE, index)
                : PhiloxReference.uniformDouble(ROOT, NODE, index);
    }

    /** A value as the op's type holds it. */
    private static double typed(DataType type, double value) {
        return type == DataType.FLOAT ? (float) value : value;
    }

    // Arithmetic in the op's type. Rounding the double result of two floats to float is the float
    // result for +, -, * and / (53 >= 2 * 24 + 2).
    private static double add(DataType type, double a, double b) {
        return typed(type, a + b);
    }

    private static double subtract(DataType type, double a, double b) {
        return typed(type, a - b);
    }

    private static double multiply(DataType type, double a, double b) {
        return typed(type, a * b);
    }

    private static double divide(DataType type, double a, double b) {
        return typed(type, a / b);
    }

    private static double ulp(DataType type, double value) {
        return type == DataType.FLOAT ? Math.ulp((float) value) : Math.ulp(value);
    }

    private static void assertExact(String op, DataType type, double[] expected, INDArray actual) {
        double[] values = actual.data().asDouble();
        assertEquals(expected.length, values.length, op + " " + type + " length");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], values[i], 0.0, op + " " + type + " index " + i);
        }
    }

    private static void assertWithin(String op, DataType type, double[] expected, double[] tolerance, INDArray actual) {
        double[] values = actual.data().asDouble();
        assertEquals(expected.length, values.length, op + " " + type + " length");
        for (int i = 0; i < expected.length; i++) {
            if (Double.isNaN(expected[i])) continue;
            assertEquals(expected[i], values[i], tolerance[i], op + " " + type + " index " + i);
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
