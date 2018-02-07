package org.deeplearning4j.earlystopping.scorecalc;

import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class AutoencoderScoreCalculator extends BaseScoreCalculator<MultiLayerNetwork> {

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
    protected INDArray output(MultiLayerNetwork network, INDArray input) {
        AutoEncoder ae = (AutoEncoder) network.getLayer(0);

        INDArray encode = ae.encode(input, false);
        return ae.decode(encode);
    }

    @Override
    protected double scoreMinibatch(MultiLayerNetwork network, INDArray features, INDArray labels, INDArray output) {
        evaluation.eval(features, output);
        return 0.0;
    }

    @Override
    protected double finalScore(double scoreSum, int minibatchCount, int exampleCount) {
        return evaluation.scoreForMetric(metric);
    }
}
