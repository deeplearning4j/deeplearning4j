package org.deeplearning4j.earlystopping.scorecalc;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public abstract class BaseMLNScoreCalculator extends BaseScoreCalculator<MultiLayerNetwork> {


    protected BaseMLNScoreCalculator(DataSetIterator iterator) {
        super(iterator);
    }

    @Override
    protected INDArray output(MultiLayerNetwork network, INDArray input) {
        return network.output(input, false);
    }
}
