package org.deeplearning4j.nn.conf.layers.variational;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationHardSigmoid;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.LessThan;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.OldLessThan;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Bernoulli reconstruction distribution for variational autoencoder.<br>
 * Outputs are modelled by a Bernoulli distribution - i.e., the Bernoulli distribution should be used for binary data (all
 * values 0 or 1); the VAE models the probability of the output being 0 or 1.<br>
 * Consequently, the sigmoid activation function should be used to bound activations to the range of 0 to 1. Activation
 * functions that do not produce outputs in the range of 0 to 1 (including relu, tanh, and many others) should be avoided.
 *
 * @author Alex Black
 */
@Slf4j
@Data
public class BernoulliReconstructionDistribution implements ReconstructionDistribution {

    private final IActivation activationFn;

    /**
     * Create a BernoulliReconstructionDistribution with the default Sigmoid activation function
     */
    public BernoulliReconstructionDistribution() {
        this("sigmoid");
    }

    /**
     * @param activationFn    Activation function. Sigmoid generally; must be bounded in range 0 to 1
     * @deprecated Use {@link #BernoulliReconstructionDistribution(Activation)}
     */
    @Deprecated
    public BernoulliReconstructionDistribution(String activationFn) {
        this(Activation.fromString(activationFn));
    }

    /**
     * @param activationFn    Activation function. Sigmoid generally; must be bounded in range 0 to 1
     */
    public BernoulliReconstructionDistribution(Activation activationFn) {
        this(activationFn.getActivationFunction());
    }

    /**
     * @param activationFn    Activation function. Sigmoid generally; must be bounded in range 0 to 1
     */
    public BernoulliReconstructionDistribution(IActivation activationFn) {
        this.activationFn = activationFn;
        if (!(activationFn instanceof ActivationSigmoid) && !(activationFn instanceof ActivationHardSigmoid)) {
            log.warn("Using BernoulliRecontructionDistribution with activation function \"" + activationFn + "\"."
                            + " Using sigmoid/hard sigmoid is recommended to bound probabilities in range 0 to 1");
        }
    }

    @Override
    public boolean hasLossFunction() {
        return false;
    }

    @Override
    public int distributionInputSize(int dataSize) {
        return dataSize;
    }

    @Override
    public double negLogProbability(INDArray x, INDArray preOutDistributionParams, boolean average) {
        INDArray logProb = calcLogProbArray(x, preOutDistributionParams);

        if (average) {
            return -logProb.sumNumber().doubleValue() / x.size(0);
        } else {
            return -logProb.sumNumber().doubleValue();
        }
    }

    @Override
    public INDArray exampleNegLogProbability(INDArray x, INDArray preOutDistributionParams) {
        INDArray logProb = calcLogProbArray(x, preOutDistributionParams);

        return logProb.sum(1).negi();
    }

    private INDArray calcLogProbArray(INDArray x, INDArray preOutDistributionParams) {
        INDArray output = preOutDistributionParams.dup();
        activationFn.getActivation(output, false);

        INDArray logOutput = Transforms.log(output, true);
        INDArray log1SubOut = Transforms.log(output.rsubi(1.0), false);

        //For numerical stability: if output = 0, then log(output) == -infinity
        //then x * log(output) = NaN, but lim(x->0, output->0)[ x * log(output) ] == 0
        // therefore: want 0*log(0) = 0, NOT 0*log(0) = NaN by default
        BooleanIndexing.replaceWhere(logOutput, 0.0, Conditions.isInfinite()); //log(out)= +/- inf -> x == 0.0 -> 0 * log(0) = 0
        BooleanIndexing.replaceWhere(log1SubOut, 0.0, Conditions.isInfinite()); //log(out)= +/- inf -> x == 0.0 -> 0 * log(0) = 0
        return logOutput.muli(x).addi(x.rsub(1.0).muli(log1SubOut));
    }

    @Override
    public INDArray gradient(INDArray x, INDArray preOutDistributionParams) {
        INDArray output = preOutDistributionParams.dup();
        activationFn.getActivation(output, true);

        INDArray diff = x.sub(output);
        INDArray outOneMinusOut = output.rsub(1.0).muli(output);

        INDArray grad = diff.divi(outOneMinusOut);
        grad = activationFn.backprop(preOutDistributionParams.dup(), grad).getFirst();

        //Issue: if output == 0 or output == 1, then (assuming sigmoid output or similar)
        //sigmaPrime == 0, sigmaPrime * (x-out) / (out*(1-out)) == 0 * (x-out) / 0 -> 0/0 -> NaN. But taking limit, we want
        //0*(x-out)/0 == 0 -> implies 0 gradient at the far extremes (0 or 1) of the output
        BooleanIndexing.replaceWhere(grad, 0.0, Conditions.isNan());
        return grad.negi();
    }

    @Override
    public INDArray generateRandom(INDArray preOutDistributionParams) {
        INDArray p = preOutDistributionParams.dup();
        activationFn.getActivation(p, false);

        INDArray rand = Nd4j.rand(p.shape());
        //Can simply randomly sample by looking where values are < p...
        //i.e., sample = 1 if randNum < p, 0 otherwise

        INDArray out = Nd4j.createUninitialized(p.shape());

        Nd4j.getExecutioner().execAndReturn(new OldLessThan(rand, p, out, p.length()));
        return out;
    }

    @Override
    public INDArray generateAtMean(INDArray preOutDistributionParams) {
        //mean value for bernoulli: same as probability parameter...
        //Obviously we can't produce exactly the mean value - bernoulli should produce only {0,1} values
        //but returning the actual mean value is more useful
        INDArray p = preOutDistributionParams.dup();
        activationFn.getActivation(p, false);

        return p;
    }

    @Override
    public String toString() {
        return "BernoulliReconstructionDistribution(afn=" + activationFn + ")";
    }
}
