package org.deeplearning4j.earlystopping.scorecalc;

import org.deeplearning4j.earlystopping.scorecalc.base.BaseScoreCalculator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Score calculator for variational autoencoder reconstruction probability or reconstruction log probability for a
 * MultiLayerNetwork or ComputationGraph. VariationalAutoencoder layer must be first layer in the network<br>
 * See {@link VariationalAutoencoder#reconstructionProbability(INDArray, int)} for more details
 *
 * @author Alex Black
 */
public class VAEReconProbScoreCalculator extends BaseScoreCalculator<Model> {

    protected final int reconstructionProbNumSamples;
    protected final boolean logProb;
    protected final boolean average;

    /**
     * Constructor for average reconstruction probability
     *
     * @param iterator Iterator
     * @param reconstructionProbNumSamples Number of samples. See {@link VariationalAutoencoder#reconstructionProbability(INDArray, int)}
     *                                    for details
     * @param logProb If true: calculate (negative) log probability. False: probability
     */
    public VAEReconProbScoreCalculator(DataSetIterator iterator, int reconstructionProbNumSamples, boolean logProb) {
        this(iterator, reconstructionProbNumSamples, logProb, true);
    }

    /**
     * Constructor for reconstruction probability
     *
     * @param iterator Iterator
     * @param reconstructionProbNumSamples Number of samples. See {@link VariationalAutoencoder#reconstructionProbability(INDArray, int)}
     *                                    for details
     * @param logProb If true: calculate (negative) log probability. False: probability
     * @param average If true: return average (log) probability. False: sum of log probability.
     *
     */
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
    protected INDArray output(Model network, INDArray input, INDArray fMask, INDArray lMask) {
        return null;    //Not used
    }

    @Override
    protected INDArray[] output(Model network, INDArray[] input, INDArray[] fMask, INDArray[] lMask) {
        return null;    //Not used
    }

    @Override
    protected double scoreMinibatch(Model net, INDArray features, INDArray labels, INDArray fMask,
                                    INDArray lMask, INDArray output) {
        Layer l;
        if(net instanceof MultiLayerNetwork) {
            MultiLayerNetwork network = (MultiLayerNetwork)net;
            l = network.getLayer(0);
        } else {
            ComputationGraph network = (ComputationGraph)net;
            l = network.getLayer(0);
        }

        if(!(l instanceof VariationalAutoencoder)){
            throw new UnsupportedOperationException("Can only score networks with VariationalAutoencoder layers as first layer -" +
                    " got " + l.getClass().getSimpleName());
        }
        VariationalAutoencoder vae = (VariationalAutoencoder)l;
        //Reconstruction prob
        if(logProb){
            return -vae.reconstructionLogProbability(features, reconstructionProbNumSamples).sumNumber().doubleValue();
        } else {
            return vae.reconstructionProbability(features, reconstructionProbNumSamples).sumNumber().doubleValue();
        }
    }

    @Override
    protected double scoreMinibatch(Model network, INDArray[] features, INDArray[] labels, INDArray[] fMask, INDArray[] lMask, INDArray[] output) {
        return 0;
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
