package org.deeplearning4j.nn.layers.variational;


import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.ReconstructionDistribution;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static junit.framework.TestCase.fail;
import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 25/11/2016.
 */
@Slf4j
public class TestReconstructionDistributions {

    @Test
    public void testGaussianLogProb() {
        Nd4j.getRandom().setSeed(12345);

        int inputSize = 4;
        int[] mbs = new int[]{1, 2, 5};

        for (boolean average : new boolean[]{true, false}) {
            for (int minibatch : mbs) {

                INDArray x = Nd4j.rand(minibatch, inputSize);
                INDArray mean = Nd4j.randn(minibatch, inputSize);
                INDArray logStdevSquared = Nd4j.rand(minibatch, inputSize).subi(0.5);

                INDArray distributionParams = Nd4j.createUninitialized(new int[]{minibatch, 2 * inputSize});
                distributionParams.get(NDArrayIndex.all(), NDArrayIndex.interval(0, inputSize)).assign(mean);
                distributionParams.get(NDArrayIndex.all(), NDArrayIndex.interval(inputSize, 2 * inputSize)).assign(logStdevSquared);

                ReconstructionDistribution dist = new GaussianReconstructionDistribution("identity");

                double logProb = dist.logProbability(x, distributionParams, average);

                //Calculate the same thing, but using Apache Commons math

                double logProbSum = 0.0;
                for (int i = 0; i < minibatch; i++) {
                    for (int j = 0; j < inputSize; j++) {
                        double mu = mean.getDouble(i, j);
                        double logSigma2 = logStdevSquared.getDouble(i, j);
                        double sigma = Math.sqrt(Math.exp(logSigma2));
                        NormalDistribution nd = new NormalDistribution(mu, sigma);

                        double xVal = x.getDouble(i, j);
                        double thisLogProb = nd.logDensity(xVal);
                        logProbSum += thisLogProb;
                    }
                }

                double expLogProb;
                if (average) {
                    expLogProb = logProbSum / minibatch;
                } else {
                    expLogProb = logProbSum;
                }


                System.out.println(expLogProb + "\t" + logProb + "\t" + (logProb / expLogProb));
                assertEquals(expLogProb, logProb, 1e-6);
            }
        }
    }

    @Test
    public void testBernoulliLogProb() {
        Nd4j.getRandom().setSeed(12345);

        int inputSize = 4;
        int[] mbs = new int[]{1, 2, 5};

        Random r = new Random(12345);

        for (boolean average : new boolean[]{true, false}) {
            for (int minibatch : mbs) {

                INDArray x = Nd4j.zeros(minibatch, inputSize);
                for (int i = 0; i < minibatch; i++) {
                    for (int j = 0; j < inputSize; j++) {
                        x.putScalar(i, j, r.nextInt(2));
                    }
                }

                INDArray distributionParams = Nd4j.rand(minibatch, inputSize).muli(2).subi(1);  //i.e., pre-sigmoid prob
                INDArray prob = Transforms.sigmoid(distributionParams, true);

                ReconstructionDistribution dist = new BernoulliReconstructionDistribution("sigmoid");

                double logProb = dist.logProbability(x, distributionParams, average);

                //Calculate the same thing, but using Apache Commons math

                double logProbSum = 0.0;
                for (int i = 0; i < minibatch; i++) {
                    for (int j = 0; j < inputSize; j++) {
                        double p = prob.getDouble(i, j);

                        BinomialDistribution binomial = new BinomialDistribution(1, p);      //Bernoulli is a special case of binomial

                        double xVal = x.getDouble(i, j);
                        double thisLogProb = binomial.logProbability((int) xVal);
                        logProbSum += thisLogProb;
                    }
                }

                double expLogProb;
                if (average) {
                    expLogProb = logProbSum / minibatch;
                } else {
                    expLogProb = logProbSum;
                }

//                System.out.println(x);

                System.out.println(expLogProb + "\t" + logProb + "\t" + (logProb / expLogProb));
                assertEquals(expLogProb, logProb, 1e-6);
            }
        }
    }

    @Test
    public void gradientCheckReconstructionDistributions() {
        double eps = 1e-6;
        double maxRelError = 1e-7;
        double minAbsoluteError = 1e-9;

        Nd4j.getRandom().setSeed(12345);

        int inputSize = 4;
        int[] mbs = new int[]{1, 3};

        Random r = new Random(12345);

        ReconstructionDistribution[] distributions = new ReconstructionDistribution[]{
                new GaussianReconstructionDistribution("identity"),
                new GaussianReconstructionDistribution("tanh"),
                new BernoulliReconstructionDistribution("sigmoid")
        };


        List<String> passes = new ArrayList<>();
        List<String> failures = new ArrayList<>();
        for (ReconstructionDistribution rd : distributions) {
            for (int minibatch : mbs) {


                INDArray x;
                INDArray distributionParams;
                if (rd instanceof GaussianReconstructionDistribution) {
                    distributionParams = Nd4j.rand(minibatch, inputSize * 2).muli(2).subi(1);
                    x = Nd4j.rand(minibatch, inputSize);
                } else {
                    distributionParams = Nd4j.rand(minibatch, inputSize).muli(2).subi(1);
                    x = Nd4j.zeros(minibatch, inputSize);
                    for (int i = 0; i < minibatch; i++) {
                        for (int j = 0; j < inputSize; j++) {
                            x.putScalar(i, j, r.nextInt(2));
                        }
                    }
                }

                INDArray gradient = rd.gradient(x, distributionParams);

                String testName = "minibatch = " + minibatch + ", size = " + inputSize + ", Distribution = " + rd;
                System.out.println("\n\n***** Starting test: " + testName + "*****");

                int totalFailureCount = 0;
                for (int i = 0; i < distributionParams.size(1); i++) {
                    for (int j = 0; j < distributionParams.size(0); j++) {
                        double initial = distributionParams.getDouble(j, i);
                        distributionParams.putScalar(j, i, initial + eps);
                        double scorePlus = rd.logProbability(x, distributionParams, false);
                        distributionParams.putScalar(j, i, initial - eps);
                        double scoreMinus = rd.logProbability(x, distributionParams, false);
                        distributionParams.putScalar(j, i, initial);

                        double numericalGrad = (scorePlus - scoreMinus) / (2.0 * eps);
                        double backpropGrad = gradient.getDouble(j, i);

                        double relError = Math.abs(numericalGrad - backpropGrad) / (Math.abs(numericalGrad) + Math.abs(backpropGrad));
                        double absError = Math.abs(backpropGrad - numericalGrad);

                        if (relError > maxRelError || Double.isNaN(relError)) {
                            if (absError < minAbsoluteError) {
                                log.info("Input (" + j + "," + i + ") passed: grad= " + backpropGrad + ", numericalGrad= " + numericalGrad
                                        + ", relError= " + relError + "; absolute error = " + absError + " < minAbsoluteError = " + minAbsoluteError);
                            } else {
                                log.info("Input (" + j + "," + i + ") FAILED: grad= " + backpropGrad + ", numericalGrad= " + numericalGrad
                                        + ", relError= " + relError + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus);
                                totalFailureCount++;
                            }
                        } else {
                            log.info("Input (" + j + "," + i + ") passed: grad= " + backpropGrad + ", numericalGrad= " + numericalGrad
                                    + ", relError= " + relError);
                        }
                    }
                }


                if(totalFailureCount > 0){
                    failures.add(testName);
                } else {
                    passes.add(testName);
                }

            }
        }

        System.out.println("\n\n\n +++++ Test Passes +++++");
        for(String s : passes){
            System.out.println(s);
        }

        System.out.println("\n\n\n +++++ Test Faliures +++++");
        for(String s : failures){
            System.out.println(s);
        }

        assertEquals(0, failures.size());
    }
}
