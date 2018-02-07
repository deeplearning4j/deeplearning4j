package org.deeplearning4j.earlystopping.scorecalc;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.EvaluationAveraging;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class ClassificationScoreCalculator extends BaseIEvaluationScoreCalculator<MultiLayerNetwork, Evaluation> {

    public enum Metric {ACCURACY, F1, PRECISION, RECALL, GMEASURE, MCC}

    protected final Metric metric;

    public ClassificationScoreCalculator(Metric metric, DataSetIterator iterator){
        super(iterator);
        this.metric = metric;
    }

    @Override
    protected Evaluation newEval() {
        return new Evaluation();
    }

    @Override
    protected double finalScore(Evaluation e) {
        switch (metric){
            case ACCURACY:
                return e.accuracy();
            case F1:
                return e.f1();
            case PRECISION:
                return e.precision();
            case RECALL:
                return e.recall();
            case GMEASURE:
                return e.gMeasure(EvaluationAveraging.Macro);
            case MCC:
                return e.matthewsCorrelation(EvaluationAveraging.Macro);
            default:
                throw new IllegalStateException("Unknown metric: " + metric);
        }
    }



}
