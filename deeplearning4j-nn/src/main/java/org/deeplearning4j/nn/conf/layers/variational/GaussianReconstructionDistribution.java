package org.deeplearning4j.nn.conf.layers.variational;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by Alex on 25/11/2016.
 */
public class GaussianReconstructionDistribution implements ReconstructionDistribution {
    @Override
    public int distributionParamCount(int dataSize) {
        return 2*dataSize;
    }

    @Override
    public double logProbability(INDArray x, INDArray preOut) {
        return 0;
    }

    @Override
    public INDArray gradient(INDArray x, INDArray preOut) {
        return null;
    }
}
