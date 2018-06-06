package org.deeplearning4j.earlystopping.scorecalc;

import org.deeplearning4j.earlystopping.scorecalc.base.BaseIEvaluationScoreCalculator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Score function for evaluating a MultiLayerNetwork according to an evaluation metric ({@link Evaluation.Metric} such
 * as accuracy, F1 score, etc.
 * Used for both MultiLayerNetwork and ComputationGraph
 *
 * @author Alex Black
 */
public class ClassificationScoreCalculator extends BaseIEvaluationScoreCalculator<Model, Evaluation> {

    protected final Evaluation.Metric metric;

    public ClassificationScoreCalculator(Evaluation.Metric metric, DataSetIterator iterator){
        super(iterator);
        this.metric = metric;
    }

    public ClassificationScoreCalculator(Evaluation.Metric metric, MultiDataSetIterator iterator){
        super(iterator);
        this.metric = metric;
    }

    @Override
    protected Evaluation newEval() {
        return new Evaluation();
    }

    @Override
    protected double finalScore(Evaluation e) {
        return e.scoreForMetric(metric);
    }
}
