package org.deeplearning4j.earlystopping.scorecalc;

import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Calculate the regression score of the network, using the specified metric
 *
 *
 * @author Alex Black
 */
public class RegressionScoreCalculator extends BaseIEvaluationScoreCalculator<MultiLayerNetwork, RegressionEvaluation> {

    public enum Metric { MSE, MAE, RMSE, RSE, PC, R2 };

    protected final Metric metric;

    public RegressionScoreCalculator(Metric metric, DataSetIterator iterator){
        super(iterator);
        this.metric = metric;
    }


    @Override
    protected RegressionEvaluation newEval() {
        return new RegressionEvaluation();
    }

    @Override
    protected double finalScore(RegressionEvaluation eval) {
        switch (metric){
            case MSE:
                return eval.averageMeanSquaredError();
            case MAE:
                return eval.averageMeanAbsoluteError();
            case RMSE:
                return eval.averagerootMeanSquaredError();
            case RSE:
                return eval.averagerelativeSquaredError();
            case PC:
                return eval.averagePearsonCorrelation();
            case R2:
                return eval.averageRSquared();
            default:
                throw new IllegalStateException("Unknown metric: " + metric);
        }
    }

}
