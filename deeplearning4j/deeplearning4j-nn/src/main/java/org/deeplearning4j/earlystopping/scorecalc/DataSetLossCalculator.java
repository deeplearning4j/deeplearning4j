package org.deeplearning4j.earlystopping.scorecalc;

import org.deeplearning4j.earlystopping.scorecalc.base.BaseScoreCalculator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/** Given a DataSetIterator: calculate the total loss for the model on that data set.
 * Can be used for both MultiLayerNetwork and ComputationGraph
 *
 * @author Alex Black
 */
public class DataSetLossCalculator extends BaseScoreCalculator<Model> {

    @JsonProperty
    private boolean average;

    /**
     * Calculate the score (loss function value) on a given data set (usually a test set)
     *
     * @param dataSetIterator Data set to calculate the score for
     * @param average         Whether to return the average (sum of loss / N) or just (sum of loss)
     */
    public DataSetLossCalculator(DataSetIterator dataSetIterator, boolean average) {
        super(dataSetIterator);
        this.average = average;
    }

    /**Calculate the score (loss function value) on a given data set (usually a test set)
     *
     * @param dataSetIterator Data set to calculate the score for
     * @param average Whether to return the average (sum of loss / N) or just (sum of loss)
     */
    public DataSetLossCalculator(MultiDataSetIterator dataSetIterator, boolean average) {
        super(dataSetIterator);
        this.average = average;
    }

    @Override
    public String toString() {
        return "DataSetLossCalculator(average=" + average + ")";
    }

    @Override
    protected void reset() {
        scoreSum = 0;
        minibatchCount = 0;
        exampleCount = 0;
    }

    @Override
    protected INDArray output(Model network, INDArray input, INDArray fMask, INDArray lMask) {
        return output(network, arr(input), arr(fMask), arr(lMask))[0];
    }

    @Override
    protected INDArray[] output(Model network, INDArray[] input, INDArray[] fMask, INDArray[] lMask) {
        if(network instanceof MultiLayerNetwork){
            INDArray out = ((MultiLayerNetwork) network).output(input[0], false, get0(fMask), get0(lMask));
            return new INDArray[]{out};
        } else if(network instanceof ComputationGraph){
            return ((ComputationGraph) network).output(false, input, fMask, lMask);
        } else {
            throw new RuntimeException("Unknown model type: " + network.getClass());
        }
    }

    @Override
    protected double scoreMinibatch(Model network, INDArray[] features, INDArray[] labels, INDArray[] fMask, INDArray[] lMask, INDArray[] output) {
        if(network instanceof MultiLayerNetwork){
            return ((MultiLayerNetwork) network).score(new DataSet(get0(features), get0(labels), get0(fMask), get0(lMask)), false)
                    * features[0].size(0);
        } else if(network instanceof ComputationGraph){
            return ((ComputationGraph) network).score(new MultiDataSet(features, labels, fMask, lMask))
                    * features[0].size(0);
        } else {
            throw new RuntimeException("Unknown model type: " + network.getClass());
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

    @Override
    public boolean minimizeScore() {
        return true;    //Minimize loss
    }
}
