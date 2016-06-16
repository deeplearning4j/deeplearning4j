package org.deeplearning4j.spark.impl.vanilla;

import lombok.Data;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by Alex on 14/06/2016.
 */
@Data
public class VanillaTrainingResult implements TrainingResult {

    private final INDArray parameters;
    private final Updater updater;
    private final double score;
    private SparkTrainingStats sparkTrainingStats;

    public VanillaTrainingResult(INDArray parameters, Updater updater, double score) {
        this(parameters, updater, score, null);
    }

    public VanillaTrainingResult(INDArray parameters, Updater updater, double score, SparkTrainingStats sparkTrainingStats) {
        this.parameters = parameters;
        this.updater = updater;
        this.score = score;
        this.sparkTrainingStats = sparkTrainingStats;
    }

    @Override
    public void setStats(SparkTrainingStats sparkTrainingStats) {
        this.sparkTrainingStats = sparkTrainingStats;
    }
}
