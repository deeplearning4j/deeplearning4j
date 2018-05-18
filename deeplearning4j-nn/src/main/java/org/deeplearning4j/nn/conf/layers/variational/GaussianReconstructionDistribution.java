package org.deeplearning4j.nn.conf.layers.variational;

import lombok.Data;
import lombok.val;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Gaussian reconstruction distribution for variational autoencoder.<br>
 * Outputs are modelled by a Gaussian distribution, with the mean and variances (diagonal covariance matrix) for each
 * output determined by the network forward pass.<br>
 * <p>
 * Specifically, the GaussianReconstructionDistribution models mean and log(stdev^2). This parameterization gives log(1) = 0,
 * and inputs can be in range (-infinity,infinity). Other parameterizations for variance are of course possible but may be
 * problematic with respect to the average pre-activation function values and activation function ranges.<br>
 * For activation functions, identity and perhaps tanh are typical - though tanh (unlike identity) implies a minimum/maximum
 * possible value for mean and log variance. Asymmetric activation functions such as sigmoid or relu should be avoided.
 *
 * @author Alex Black
 */
@Data
public class GaussianReconstructionDistribution implements ReconstructionDistribution {

    private static final double NEG_HALF_LOG_2PI = -0.5 * Math.log(2 * Math.PI);

    private final IActivation activationFn;

    /**
     * Create a GaussianReconstructionDistribution with the default identity activation function.
     */
    public GaussianReconstructionDistribution() {
        this(Activation.IDENTITY);
    }

    /**
     * @param activationFn    Activation function for the reconstruction distribution. Typically identity or tanh.
     */
    public GaussianReconstructionDistribution(Activation activationFn) {
        this(activationFn.getActivationFunction());
    }

    /**
     * @param activationFn    Activation function for the reconstruction distribution. Typically identity or tanh.
     */
    public GaussianReconstructionDistribution(IActivation activationFn) {
        this.activationFn = activationFn;
    }

    @Override
    public boolean hasLossFunction() {
        return false;
    }

    @Override
    public int distributionInputSize(int dataSize) {
        return 2 * dataSize;
    }

    @Override
    public double negLogProbability(INDArray x, INDArray preOutDistributionParams, boolean average) {
        val size = preOutDistributionParams.size(1) / 2;

        INDArray[] logProbArrays = calcLogProbArrayExConstants(x, preOutDistributionParams);
        double logProb = x.size(0) * size * NEG_HALF_LOG_2PI - 0.5 * logProbArrays[0].sumNumber().doubleValue()
                        - logProbArrays[1].sumNumber().doubleValue();

        if (average) {
            return -logProb / x.size(0);
        } else {
            return -logProb;
        }
    }

    @Override
    public INDArray exampleNegLogProbability(INDArray x, INDArray preOutDistributionParams) {
        val size = preOutDistributionParams.size(1) / 2;

        INDArray[] logProbArrays = calcLogProbArrayExConstants(x, preOutDistributionParams);

        return logProbArrays[0].sum(1).muli(0.5).subi(size * NEG_HALF_LOG_2PI).addi(logProbArrays[1].sum(1));
    }

    private INDArray[] calcLogProbArrayExConstants(INDArray x, INDArray preOutDistributionParams) {
        INDArray output = preOutDistributionParams.dup();
        activationFn.getActivation(output, false);

        val size = output.size(1) / 2;
        INDArray mean = output.get(NDArrayIndex.all(), NDArrayIndex.interval(0, size));
        INDArray logStdevSquared = output.get(NDArrayIndex.all(), NDArrayIndex.interval(size, 2 * size));

        INDArray sigmaSquared = Transforms.exp(logStdevSquared, true);
        INDArray lastTerm = x.sub(mean);
        lastTerm.muli(lastTerm);
        lastTerm.divi(sigmaSquared).divi(2);

        return new INDArray[] {logStdevSquared, lastTerm};
    }

    @Override
    public INDArray gradient(INDArray x, INDArray preOutDistributionParams) {
        INDArray output = preOutDistributionParams.dup();
        activationFn.getActivation(output, true);

        val size = output.size(1) / 2;
        INDArray mean = output.get(NDArrayIndex.all(), NDArrayIndex.interval(0, size));
        INDArray logStdevSquared = output.get(NDArrayIndex.all(), NDArrayIndex.interval(size, 2 * size));

        INDArray sigmaSquared = Transforms.exp(logStdevSquared, true);

        INDArray xSubMean = x.sub(mean);
        INDArray xSubMeanSq = xSubMean.mul(xSubMean);

        INDArray dLdmu = xSubMean.divi(sigmaSquared);

        INDArray sigma = Transforms.sqrt(sigmaSquared, true);
        INDArray sigma3 = Transforms.pow(sigmaSquared, 3.0 / 2);

        INDArray dLdsigma = sigma.rdiv(-1).addi(xSubMeanSq.divi(sigma3));
        INDArray dLdlogSigma2 = sigma.divi(2).muli(dLdsigma);

        INDArray dLdx = Nd4j.createUninitialized(output.shape());
        dLdx.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.interval(0, size)}, dLdmu);
        dLdx.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.interval(size, 2 * size)}, dLdlogSigma2);
        dLdx.negi();

        //dL/dz
        return activationFn.backprop(preOutDistributionParams.dup(), dLdx).getFirst();
    }

    @Override
    public INDArray generateRandom(INDArray preOutDistributionParams) {
        INDArray output = preOutDistributionParams.dup();
        activationFn.getActivation(output, true);

        val size = output.size(1) / 2;
        INDArray mean = output.get(NDArrayIndex.all(), NDArrayIndex.interval(0, size));
        INDArray logStdevSquared = output.get(NDArrayIndex.all(), NDArrayIndex.interval(size, 2 * size));

        INDArray sigma = Transforms.exp(logStdevSquared, true);
        Transforms.sqrt(sigma, false);

        INDArray e = Nd4j.randn(sigma.shape());
        return e.muli(sigma).addi(mean); //mu + sigma * N(0,1) ~ N(mu,sigma^2)
    }

    @Override
    public INDArray generateAtMean(INDArray preOutDistributionParams) {
        val size = preOutDistributionParams.size(1) / 2;
        INDArray mean = preOutDistributionParams.get(NDArrayIndex.all(), NDArrayIndex.interval(0, size)).dup();
        activationFn.getActivation(mean, false);

        return mean;
    }

    @Override
    public String toString() {
        return "GaussianReconstructionDistribution(afn=" + activationFn + ")";
    }
}
