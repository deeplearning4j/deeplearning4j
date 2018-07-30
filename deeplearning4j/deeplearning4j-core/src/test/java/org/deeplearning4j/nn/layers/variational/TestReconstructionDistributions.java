/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.layers.variational;


import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.commons.math3.distribution.ExponentialDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.ExponentialReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.ReconstructionDistribution;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * Created by Alex on 25/11/2016.
 */
@Slf4j
public class TestReconstructionDistributions extends BaseDL4JTest {

    @Test
    public void testGaussianLogProb() {
        Nd4j.getRandom().setSeed(12345);

        int inputSize = 4;
        int[] mbs = new int[] {1, 2, 5};

        for (boolean average : new boolean[] {true, false}) {
            for (int minibatch : mbs) {

                INDArray x = Nd4j.rand(minibatch, inputSize);
                INDArray mean = Nd4j.randn(minibatch, inputSize);
                INDArray logStdevSquared = Nd4j.rand(minibatch, inputSize).subi(0.5);

                INDArray distributionParams = Nd4j.createUninitialized(new int[] {minibatch, 2 * inputSize});
                distributionParams.get(NDArrayIndex.all(), NDArrayIndex.interval(0, inputSize)).assign(mean);
                distributionParams.get(NDArrayIndex.all(), NDArrayIndex.interval(inputSize, 2 * inputSize))
                                .assign(logStdevSquared);

                ReconstructionDistribution dist = new GaussianReconstructionDistribution(Activation.IDENTITY);

                double negLogProb = dist.negLogProbability(x, distributionParams, average);

                INDArray exampleNegLogProb = dist.exampleNegLogProbability(x, distributionParams);
                assertArrayEquals(new long[] {minibatch, 1}, exampleNegLogProb.shape());

                //Calculate the same thing, but using Apache Commons math

                double logProbSum = 0.0;
                for (int i = 0; i < minibatch; i++) {
                    double exampleSum = 0.0;
                    for (int j = 0; j < inputSize; j++) {
                        double mu = mean.getDouble(i, j);
                        double logSigma2 = logStdevSquared.getDouble(i, j);
                        double sigma = Math.sqrt(Math.exp(logSigma2));
                        NormalDistribution nd = new NormalDistribution(mu, sigma);

                        double xVal = x.getDouble(i, j);
                        double thisLogProb = nd.logDensity(xVal);
                        logProbSum += thisLogProb;
                        exampleSum += thisLogProb;
                    }
                    assertEquals(-exampleNegLogProb.getDouble(i), exampleSum, 1e-6);
                }

                double expNegLogProb;
                if (average) {
                    expNegLogProb = -logProbSum / minibatch;
                } else {
                    expNegLogProb = -logProbSum;
                }


                //                System.out.println(expLogProb + "\t" + logProb + "\t" + (logProb / expLogProb));
                assertEquals(expNegLogProb, negLogProb, 1e-6);


                //Also: check random sampling...
                int count = minibatch * inputSize;
                INDArray arr = Nd4j.linspace(-3, 3, count).reshape(minibatch, inputSize);
                INDArray sampleMean = dist.generateAtMean(arr);
                INDArray sampleRandom = dist.generateRandom(arr);
            }
        }
    }

    @Test
    public void testBernoulliLogProb() {
        Nd4j.getRandom().setSeed(12345);

        int inputSize = 4;
        int[] mbs = new int[] {1, 2, 5};

        Random r = new Random(12345);

        for (boolean average : new boolean[] {true, false}) {
            for (int minibatch : mbs) {

                INDArray x = Nd4j.zeros(minibatch, inputSize);
                for (int i = 0; i < minibatch; i++) {
                    for (int j = 0; j < inputSize; j++) {
                        x.putScalar(i, j, r.nextInt(2));
                    }
                }

                INDArray distributionParams = Nd4j.rand(minibatch, inputSize).muli(2).subi(1); //i.e., pre-sigmoid prob
                INDArray prob = Transforms.sigmoid(distributionParams, true);

                ReconstructionDistribution dist = new BernoulliReconstructionDistribution(Activation.SIGMOID);

                double negLogProb = dist.negLogProbability(x, distributionParams, average);

                INDArray exampleNegLogProb = dist.exampleNegLogProbability(x, distributionParams);
                assertArrayEquals(new long[] {minibatch, 1}, exampleNegLogProb.shape());

                //Calculate the same thing, but using Apache Commons math

                double logProbSum = 0.0;
                for (int i = 0; i < minibatch; i++) {
                    double exampleSum = 0.0;
                    for (int j = 0; j < inputSize; j++) {
                        double p = prob.getDouble(i, j);

                        BinomialDistribution binomial = new BinomialDistribution(1, p); //Bernoulli is a special case of binomial

                        double xVal = x.getDouble(i, j);
                        double thisLogProb = binomial.logProbability((int) xVal);
                        logProbSum += thisLogProb;
                        exampleSum += thisLogProb;
                    }
                    assertEquals(-exampleNegLogProb.getDouble(i), exampleSum, 1e-6);
                }

                double expNegLogProb;
                if (average) {
                    expNegLogProb = -logProbSum / minibatch;
                } else {
                    expNegLogProb = -logProbSum;
                }

                //                System.out.println(x);

                //                System.out.println(expNegLogProb + "\t" + logProb + "\t" + (logProb / expNegLogProb));
                assertEquals(expNegLogProb, negLogProb, 1e-6);

                //Also: check random sampling...
                int count = minibatch * inputSize;
                INDArray arr = Nd4j.linspace(-3, 3, count).reshape(minibatch, inputSize);
                INDArray sampleMean = dist.generateAtMean(arr);
                INDArray sampleRandom = dist.generateRandom(arr);

                for (int i = 0; i < minibatch; i++) {
                    for (int j = 0; j < inputSize; j++) {
                        double d1 = sampleMean.getDouble(i, j);
                        double d2 = sampleRandom.getDouble(i, j);
                        assertTrue(d1 >= 0.0 || d1 <= 1.0); //Mean value - probability... could do 0 or 1 (based on most likely) but that isn't very useful...
                        assertTrue(d2 == 0.0 || d2 == 1.0);
                    }
                }
            }
        }
    }

    @Test
    public void testExponentialLogProb() {
        Nd4j.getRandom().setSeed(12345);

        int inputSize = 4;
        int[] mbs = new int[] {1, 2, 5};

        Random r = new Random(12345);

        for (boolean average : new boolean[] {true, false}) {
            for (int minibatch : mbs) {

                INDArray x = Nd4j.zeros(minibatch, inputSize);
                for (int i = 0; i < minibatch; i++) {
                    for (int j = 0; j < inputSize; j++) {
                        x.putScalar(i, j, r.nextInt(2));
                    }
                }

                INDArray distributionParams = Nd4j.rand(minibatch, inputSize).muli(2).subi(1); //i.e., pre-afn gamma
                INDArray gammas = Transforms.tanh(distributionParams, true);

                ReconstructionDistribution dist = new ExponentialReconstructionDistribution(Activation.TANH);

                double negLogProb = dist.negLogProbability(x, distributionParams, average);

                INDArray exampleNegLogProb = dist.exampleNegLogProbability(x, distributionParams);
                assertArrayEquals(new long[] {minibatch, 1}, exampleNegLogProb.shape());

                //Calculate the same thing, but using Apache Commons math

                double logProbSum = 0.0;
                for (int i = 0; i < minibatch; i++) {
                    double exampleSum = 0.0;
                    for (int j = 0; j < inputSize; j++) {
                        double gamma = gammas.getDouble(i, j);
                        double lambda = Math.exp(gamma);
                        double mean = 1.0 / lambda;

                        ExponentialDistribution exp = new ExponentialDistribution(mean); //Commons math uses mean = 1/lambda

                        double xVal = x.getDouble(i, j);
                        double thisLogProb = exp.logDensity(xVal);
                        logProbSum += thisLogProb;
                        exampleSum += thisLogProb;
                    }
                    assertEquals(-exampleNegLogProb.getDouble(i), exampleSum, 1e-6);
                }

                double expNegLogProb;
                if (average) {
                    expNegLogProb = -logProbSum / minibatch;
                } else {
                    expNegLogProb = -logProbSum;
                }

                //                System.out.println(x);

                //                System.out.println(expNegLogProb + "\t" + logProb + "\t" + (logProb / expNegLogProb));
                assertEquals(expNegLogProb, negLogProb, 1e-6);

                //Also: check random sampling...
                int count = minibatch * inputSize;
                INDArray arr = Nd4j.linspace(-3, 3, count).reshape(minibatch, inputSize);
                INDArray sampleMean = dist.generateAtMean(arr);
                INDArray sampleRandom = dist.generateRandom(arr);

                for (int i = 0; i < minibatch; i++) {
                    for (int j = 0; j < inputSize; j++) {
                        double d1 = sampleMean.getDouble(i, j);
                        double d2 = sampleRandom.getDouble(i, j);
                        assertTrue(d1 >= 0.0);
                        assertTrue(d2 >= 0.0);
                    }
                }
            }
        }
    }

    @Test
    public void gradientCheckReconstructionDistributions() {
        double eps = 1e-6;
        double maxRelError = 1e-6;
        double minAbsoluteError = 1e-9;

        Nd4j.getRandom().setSeed(12345);

        int inputSize = 4;
        int[] mbs = new int[] {1, 3};

        Random r = new Random(12345);

        ReconstructionDistribution[] distributions =
                        new ReconstructionDistribution[] {new GaussianReconstructionDistribution(Activation.IDENTITY),
                                        new GaussianReconstructionDistribution(Activation.TANH),
                                        new BernoulliReconstructionDistribution(Activation.SIGMOID),
                                        new ExponentialReconstructionDistribution(Activation.IDENTITY),
                                        new ExponentialReconstructionDistribution(Activation.TANH)};


        List<String> passes = new ArrayList<>();
        List<String> failures = new ArrayList<>();
        for (ReconstructionDistribution rd : distributions) {
            for (int minibatch : mbs) {


                INDArray x;
                INDArray distributionParams;
                if (rd instanceof GaussianReconstructionDistribution) {
                    distributionParams = Nd4j.rand(minibatch, inputSize * 2).muli(2).subi(1);
                    x = Nd4j.rand(minibatch, inputSize);
                } else if (rd instanceof BernoulliReconstructionDistribution) {
                    distributionParams = Nd4j.rand(minibatch, inputSize).muli(2).subi(1);
                    x = Nd4j.zeros(minibatch, inputSize);
                    for (int i = 0; i < minibatch; i++) {
                        for (int j = 0; j < inputSize; j++) {
                            x.putScalar(i, j, r.nextInt(2));
                        }
                    }
                } else if (rd instanceof ExponentialReconstructionDistribution) {
                    distributionParams = Nd4j.rand(minibatch, inputSize).muli(2).subi(1);
                    x = Nd4j.rand(minibatch, inputSize);
                } else {
                    throw new RuntimeException();
                }

                INDArray gradient = rd.gradient(x, distributionParams);

                String testName = "minibatch = " + minibatch + ", size = " + inputSize + ", Distribution = " + rd;
                System.out.println("\n\n***** Starting test: " + testName + "*****");

                int totalFailureCount = 0;
                for (int i = 0; i < distributionParams.size(1); i++) {
                    for (int j = 0; j < distributionParams.size(0); j++) {
                        double initial = distributionParams.getDouble(j, i);
                        distributionParams.putScalar(j, i, initial + eps);
                        double scorePlus = rd.negLogProbability(x, distributionParams, false);
                        distributionParams.putScalar(j, i, initial - eps);
                        double scoreMinus = rd.negLogProbability(x, distributionParams, false);
                        distributionParams.putScalar(j, i, initial);

                        double numericalGrad = (scorePlus - scoreMinus) / (2.0 * eps);
                        double backpropGrad = gradient.getDouble(j, i);

                        double relError = Math.abs(numericalGrad - backpropGrad)
                                        / (Math.abs(numericalGrad) + Math.abs(backpropGrad));
                        double absError = Math.abs(backpropGrad - numericalGrad);

                        if (relError > maxRelError || Double.isNaN(relError)) {
                            if (absError < minAbsoluteError) {
                                log.info("Input (" + j + "," + i + ") passed: grad= " + backpropGrad
                                                + ", numericalGrad= " + numericalGrad + ", relError= " + relError
                                                + "; absolute error = " + absError + " < minAbsoluteError = "
                                                + minAbsoluteError);
                            } else {
                                log.info("Input (" + j + "," + i + ") FAILED: grad= " + backpropGrad
                                                + ", numericalGrad= " + numericalGrad + ", relError= " + relError
                                                + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus);
                                totalFailureCount++;
                            }
                        } else {
                            log.info("Input (" + j + "," + i + ") passed: grad= " + backpropGrad + ", numericalGrad= "
                                            + numericalGrad + ", relError= " + relError);
                        }
                    }
                }


                if (totalFailureCount > 0) {
                    failures.add(testName);
                } else {
                    passes.add(testName);
                }

            }
        }

        System.out.println("\n\n\n +++++ Test Passes +++++");
        for (String s : passes) {
            System.out.println(s);
        }

        System.out.println("\n\n\n +++++ Test Faliures +++++");
        for (String s : failures) {
            System.out.println(s);
        }

        assertEquals(0, failures.size());
    }
}
