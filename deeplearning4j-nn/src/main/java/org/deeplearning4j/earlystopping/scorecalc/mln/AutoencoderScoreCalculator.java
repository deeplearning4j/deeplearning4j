package org.deeplearning4j.earlystopping.scorecalc.mln;

import org.deeplearning4j.earlystopping.scorecalc.base.BaseMLNScoreCalculator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Score function for a MultiLayerNetwork with a single {@link org.deeplearning4j.nn.conf.layers.AutoEncoder} layer.
 * Calculates the specified {@link RegressionEvaluation.Metric} on the layer's reconstructions.
 *
 * @author Alex Black
 */
public class AutoencoderScoreCalculator extends BaseMLNScoreCalculator {

    protected final RegressionEvaluation.Metric metric;
    protected RegressionEvaluation evaluation;

    public AutoencoderScoreCalculator(RegressionEvaluation.Metric metric, DataSetIterator iterator){
        super(iterator);
        this.metric = metric;
    }

    @Override
    protected void reset() {
        evaluation = new RegressionEvaluation();
    }

    @Override
    protected INDArray output(MultiLayerNetwork network, INDArray input, INDArray fMask, INDArray lMask) {
        Layer l = network.getLayer(0);
        if(!(l instanceof AutoEncoder)){
            throw new UnsupportedOperationException("Can only score networks with autoencoder layers as first layer -" +
                    " got " + l.getClass().getSimpleName());
        }
        AutoEncoder ae = (AutoEncoder) l;

        INDArray encode = ae.encode(input, false);
        return ae.decode(encode);
    }

    @Override
    protected double scoreMinibatch(MultiLayerNetwork network, INDArray features, INDArray labels, INDArray fMask,
                                    INDArray lMask, INDArray output) {
        evaluation.eval(features, output);
        return 0.0; //Not used
    }

    @Override
    protected double finalScore(double scoreSum, int minibatchCount, int exampleCount) {
        return evaluation.scoreForMetric(metric);
    }
}
