package org.deeplearning4j.earlystopping.scorecalc;

import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Score calculator for variational autoencoder reconstruction probability or reconstruction log probability
 *
 * @author Alex Black
 */
public class VAEReconProbScoreCalculator extends BaseScoreCalculator<MultiLayerNetwork> {

    protected final int reconstructionProbNumSamples;
    protected final boolean logProb;
    protected final boolean average;

    /**
     * Constructor for reconstruction probability
     *
     * @param iterator
     */
    public VAEReconProbScoreCalculator(DataSetIterator iterator, int reconstructionProbNumSamples, boolean logProb) {
        this(iterator, reconstructionProbNumSamples, logProb, true);
    }

    public VAEReconProbScoreCalculator(DataSetIterator iterator, int reconstructionProbNumSamples, boolean logProb,
                                       boolean average){
        super(iterator);
        this.reconstructionProbNumSamples = reconstructionProbNumSamples;
        this.logProb = logProb;
        this.average = average;
    }

    @Override
    protected void reset() {
        scoreSum = 0;
        minibatchCount = 0;
        exampleCount = 0;
    }

    @Override
    protected INDArray output(MultiLayerNetwork network, INDArray input) {
        return null;    //Not used
    }

    @Override
    protected double scoreMinibatch(MultiLayerNetwork network, INDArray features, INDArray labels, INDArray output) {
        Layer l = network.getLayer(0);
        if(!(l instanceof VariationalAutoencoder)){
            throw new UnsupportedOperationException("Can only score networks with VariationalAutoencoder layers as first layer -" +
                    " got " + l.getClass().getSimpleName());
        }
        VariationalAutoencoder vae = (VariationalAutoencoder)l;
        //Reconstruction prob
        if(logProb){
            return vae.reconstructionLogProbability(features, reconstructionProbNumSamples).sumNumber().doubleValue();
        } else {
            return vae.reconstructionProbability(features, reconstructionProbNumSamples).sumNumber().doubleValue();
        }
    }

    @Override
    protected double finalScore(double scoreSum, int minibatchCount, int exampleCount) {
        if(average){
            return scoreSum / exampleCount;
        } else {
            return scoreSum;
        }
    }
}
