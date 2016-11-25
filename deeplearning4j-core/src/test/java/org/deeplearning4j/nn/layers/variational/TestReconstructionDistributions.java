package org.deeplearning4j.nn.layers.variational;


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

import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 25/11/2016.
 */
public class TestReconstructionDistributions {

    @Test
    public void testGaussianLogProb(){
        Nd4j.getRandom().setSeed(12345);

        int inputSize = 4;
        int[] mbs = new int[]{1,2,5};

        for(boolean average : new boolean[]{true,false}) {
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
                if(average){
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
    public void testBernoulliLogProb(){
        Nd4j.getRandom().setSeed(12345);

        int inputSize = 4;
        int[] mbs = new int[]{1,2,5};

        Random r = new Random(12345);

        for(boolean average : new boolean[]{true,false}) {
            for (int minibatch : mbs) {

                INDArray x = Nd4j.zeros(minibatch, inputSize);
                for( int i=0; i<minibatch; i++ ){
                    for( int j=0; j<inputSize; j++ ){
                        x.putScalar(i,j,r.nextInt(2));
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
                        double p = prob.getDouble(i,j);

                        BinomialDistribution binomial = new BinomialDistribution(1,p);      //Bernoulli is a special case of binomial

                        double xVal = x.getDouble(i, j);
                        double thisLogProb = binomial.logProbability((int)xVal);
                        logProbSum += thisLogProb;
                    }
                }

                double expLogProb;
                if(average){
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
}
