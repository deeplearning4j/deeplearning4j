package org.deeplearning4j.earlystopping.scorecalc.base;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Abstract score calculator for MultiLayerNetwonk
 *
 * @author Alex Black
 */
public abstract class BaseMLNScoreCalculator extends BaseScoreCalculator<MultiLayerNetwork> {


    protected BaseMLNScoreCalculator(DataSetIterator iterator) {
        super(iterator);
    }

    @Override
    protected INDArray output(MultiLayerNetwork network, INDArray input, INDArray fMask, INDArray lMask) {
        return network.output(input, false, fMask, lMask);
    }

    @Override
    protected double scoreMinibatch(MultiLayerNetwork network, INDArray[] features, INDArray[] labels, INDArray[] fMask,
                                    INDArray[] lMask, INDArray[] output) {
        return scoreMinibatch(network, get0(features), get0(labels), get0(fMask), get0(lMask), get0(output));
    }

    @Override
    protected INDArray[] output(MultiLayerNetwork network, INDArray[] input, INDArray[] fMask, INDArray[] lMask) {
        return new INDArray[]{output(network, get0(input), get0(fMask), get0(lMask))};
    }
}
