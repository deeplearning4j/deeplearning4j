package org.deeplearning4j.earlystopping.scorecalc;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;

/** Given a DataSetIterator: calculate the total loss for the model on that data set.
 * Typically used to calculate the loss on a test set.
 * Note: For early stopping on a {@link ComputationGraph} use {@link DataSetLossCalculatorCG}
 */
public class DataSetLossCalculator implements ScoreCalculator<MultiLayerNetwork>{

    private DataSetIterator dataSetIterator;
    private boolean average;

    /**Calculate the score (loss function value) on a given data set (usually a test set)
     *
     * @param dataSetIterator Data set to calculate the score for
     * @param average Whether to return the average (sum of loss / N) or just (sum of loss)
     */
    public DataSetLossCalculator(DataSetIterator dataSetIterator, boolean average) {
        this.dataSetIterator = dataSetIterator;
        this.average = average;
    }

    @Override
    public double calculateScore(MultiLayerNetwork network) {
        dataSetIterator.reset();

        double lossSum = 0.0;
        int exCount = 0;
        while(dataSetIterator.hasNext()){
            DataSet dataSet = dataSetIterator.next();
            if(dataSet==null) break;
            int nEx = dataSet.getFeatureMatrix().size(0);
            lossSum += network.score(dataSet) * nEx;
            exCount += nEx;
        }

        if(average) return lossSum / exCount;
        else return lossSum;
    }

    @Override
    public String toString(){
        return "DataSetLossCalculator(" + dataSetIterator + ",average="+average+")";
    }
}
