package org.deeplearning4j.nn.conf.layers.variational;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
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
public class BernoulliReconstructionDistribution implements ReconstructionDistribution {

    private String activationFn;

    /**
     * Create a BernoulliReconstructionDistribution with the default Sigmoid activation function
     */
    public BernoulliReconstructionDistribution(){
        this("sigmoid");
    }

    /**
     * @param activationFn    Activation function. Sigmoid generally; must be bounded in range 0 to 1
     */
    public BernoulliReconstructionDistribution(String activationFn){
        this.activationFn = activationFn;
        if(!"sigmoid".equals(activationFn)){
            log.warn("Using BernoulliRecontructionDistribution with activation function \"" + activationFn + "\"."
                    + " Using sigmoid is recommended to bound probabilities in range 0 to 1");
        }
    }

    @Override
    public int distributionInputSize(int dataSize) {
        return dataSize;
    }

    @Override
    public double logProbability(INDArray x, INDArray preOutDistributionParams, boolean average) {
        INDArray output = preOutDistributionParams.dup();
        if(!"identity".equals(activationFn)){
            output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, output));
        }

        INDArray logOutput = Transforms.log(output, true);
        INDArray log1SubOut = Transforms.log(output.rsubi(1.0), false);

        INDArray logProb = logOutput.muli(x).addi(x.rsub(1.0).muli(log1SubOut));

        if(average){
            return logProb.sumNumber().doubleValue() / x.size(0);
        } else {
            return logProb.sumNumber().doubleValue();
        }
    }

    @Override
    public INDArray gradient(INDArray x, INDArray preOutDistributionParams) {
        INDArray output = preOutDistributionParams.dup();
        if(!"identity".equals(activationFn)){
            output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, output));
        }

        INDArray dif = x.sub(output);
        INDArray outOneMinusOut = output.rsub(1.0).muli(output);

        INDArray grad = dif.divi(outOneMinusOut);

        if(!"identity".equals(activationFn)){
            INDArray sigmaPrimeZ = Nd4j.getExecutioner().execAndReturn(
                    Nd4j.getOpFactory().createTransform(activationFn, preOutDistributionParams.dup()).derivative());
            grad.muli(sigmaPrimeZ);
        }

        return grad;
    }

    @Override
    public String toString(){
        return "BernoulliReconstructionDistribution(afn=" + activationFn + ")";
    }
}
