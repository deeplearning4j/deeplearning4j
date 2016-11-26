package org.deeplearning4j.nn.conf.layers.variational;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Created by Alex on 25/11/2016.
 */
@Slf4j
public class BernoulliReconstructionDistribution implements ReconstructionDistribution {

    private String activationFn;

    public BernoulliReconstructionDistribution(String activationFn){
        this.activationFn = activationFn;
        if(!"sigmoid".equals(activationFn)){
            log.warn("Using BernoulliRecontructionDistribution with activation function \"" + activationFn + "\"."
                    + " Using sigmoid is recommended to bound probabilities in range 0 to 1");
        }
    }

    @Override
    public int distributionParamCount(int dataSize) {
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
        return null;
    }
}
