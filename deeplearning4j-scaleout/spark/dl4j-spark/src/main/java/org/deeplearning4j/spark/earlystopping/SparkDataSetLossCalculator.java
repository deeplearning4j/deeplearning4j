package org.deeplearning4j.spark.earlystopping;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.dataset.DataSet;

/** Given a DataSetIterator: calculate the total loss for the model on that data set, using Spark.
 * Typically used to calculate the loss on a test set.
 */
public class SparkDataSetLossCalculator implements ScoreCalculator<MultiLayerNetwork> {


    private JavaRDD<DataSet> data;
    private boolean average;
    private SparkContext sc;

    /**Calculate the score (loss function value) on a given data set (usually a test set)
     *
     * @param data Data set to calculate the score for
     * @param average Whether to return the average (sum of loss / N) or just (sum of loss)
     */
    public SparkDataSetLossCalculator(JavaRDD<DataSet> data, boolean average, SparkContext sc) {
        this.data = data;
        this.average = average;
        this.sc = sc;
    }

    @Override
    public double calculateScore(MultiLayerNetwork network) {

        SparkDl4jMultiLayer net = new SparkDl4jMultiLayer(sc,network);
        return net.calculateScore(data,average);
    }

}
