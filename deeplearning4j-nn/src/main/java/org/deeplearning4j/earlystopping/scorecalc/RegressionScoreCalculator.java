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



    protected final RegressionEvaluation.Metric metric;

    public RegressionScoreCalculator(RegressionEvaluation.Metric metric, DataSetIterator iterator){
        super(iterator);
        this.metric = metric;
    }


    @Override
    protected RegressionEvaluation newEval() {
        return new RegressionEvaluation();
    }

    @Override
    protected double finalScore(RegressionEvaluation eval) {
        return eval.scoreForMetric(metric);
    }

}
