package org.deeplearning4j.spark.impl.paramavg;

import lombok.Data;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * The results (parameters, optional updaters) returned by a {@link ParameterAveragingTrainingWorker} to the
 * {@link ParameterAveragingTrainingMaster}
 *
 * @author Alex Black
 */
@Data
public class ParameterAveragingTrainingResult implements TrainingResult {

    private final INDArray parameters;
//    private final Updater updater;
//    private final ComputationGraphUpdater graphUpdater;
    private final INDArray updaterState;
    private final double score;
    private SparkTrainingStats sparkTrainingStats;

    public ParameterAveragingTrainingResult(INDArray parameters, INDArray updaterState, double score) {
        this(parameters, updaterState, score, null);
    }

    public ParameterAveragingTrainingResult(INDArray parameters, INDArray updaterState, double score, SparkTrainingStats sparkTrainingStats) {
        this.parameters = parameters;
        this.updaterState = updaterState;
        this.score = score;
        this.sparkTrainingStats = sparkTrainingStats;
    }

    @Override
    public void setStats(SparkTrainingStats sparkTrainingStats) {
        this.sparkTrainingStats = sparkTrainingStats;
    }
}
