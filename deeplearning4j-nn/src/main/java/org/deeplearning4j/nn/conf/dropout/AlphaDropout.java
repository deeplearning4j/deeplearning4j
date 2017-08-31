package org.deeplearning4j.nn.conf.dropout;

import org.nd4j.linalg.api.ndarray.INDArray;

public class AlphaDropout implements IDropout {
    @Override
    public INDArray applyDropout(INDArray inputActivations, int iteration, int epoch, boolean inPlace) {
        return null;
    }
}
