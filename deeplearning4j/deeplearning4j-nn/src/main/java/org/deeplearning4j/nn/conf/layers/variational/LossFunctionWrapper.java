package org.deeplearning4j.nn.conf.layers.variational;

import lombok.Data;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * LossFunctionWrapper allows training of a VAE model with a standard (possibly deterministic) neural network loss function
 * for the reconstruction, instead of using a {@link ReconstructionDistribution} as would normally be done with a VAE model.
 * <p>
 * Note: most functionality is supported, but clearly reconstruction log probability cannot be calculated when using
 * LossFunctionWrapper, as ILossFunction instances do not have either (a) a probabilistic interpretation, or (b) a
 * means of calculating the negative log probability.
 *
 * @author Alex Black
 */
@Data
public class LossFunctionWrapper implements ReconstructionDistribution {

    private final IActivation activationFn;
    private final ILossFunction lossFunction;

    public LossFunctionWrapper(@JsonProperty("activationFn") IActivation activationFn,
                    @JsonProperty("lossFunction") ILossFunction lossFunction) {
        this.activationFn = activationFn;
        this.lossFunction = lossFunction;
    }

    public LossFunctionWrapper(Activation activation, ILossFunction lossFunction) {
        this(activation.getActivationFunction(), lossFunction);
    }

    @Override
    public boolean hasLossFunction() {
        return true;
    }

    @Override
    public int distributionInputSize(int dataSize) {
        return dataSize;
    }

    @Override
    public double negLogProbability(INDArray x, INDArray preOutDistributionParams, boolean average) {

        //NOTE: The returned value here is NOT negative log probability, but it (the loss function value)
        // is equivalent, in terms of being something we want to minimize...

        return lossFunction.computeScore(x, preOutDistributionParams, activationFn, null, average);
    }

    @Override
    public INDArray exampleNegLogProbability(INDArray x, INDArray preOutDistributionParams) {
        return lossFunction.computeScoreArray(x, preOutDistributionParams, activationFn, null);
    }

    @Override
    public INDArray gradient(INDArray x, INDArray preOutDistributionParams) {
        return lossFunction.computeGradient(x, preOutDistributionParams, activationFn, null);
    }

    @Override
    public INDArray generateRandom(INDArray preOutDistributionParams) {
        //Loss functions: not probabilistic -> deterministic output
        return generateAtMean(preOutDistributionParams);
    }

    @Override
    public INDArray generateAtMean(INDArray preOutDistributionParams) {
        //Loss functions: not probabilistic -> not random
        INDArray out = preOutDistributionParams.dup();
        return activationFn.getActivation(out, true);
    }

    @Override
    public String toString() {
        return "LossFunctionWrapper(afn=" + activationFn + "," + lossFunction + ")";
    }
}
