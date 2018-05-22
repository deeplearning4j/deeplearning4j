package org.deeplearning4j.earlystopping.scorecalc.base;

import lombok.AllArgsConstructor;
import org.deeplearning4j.datasets.iterator.MultiDataSetWrapperIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Base score function based on an IEvaluation instance. Used for both MultiLayerNetwork and ComputationGraph
 *
 * @param <T> Type of model
 * @param <U> Type of evaluation
 */
public abstract class BaseIEvaluationScoreCalculator<T extends Model, U extends IEvaluation> implements ScoreCalculator<T> {

    protected MultiDataSetIterator iterator;
    protected DataSetIterator iter;

    protected BaseIEvaluationScoreCalculator(MultiDataSetIterator iterator){
        this.iterator = iterator;
    }

    protected BaseIEvaluationScoreCalculator(DataSetIterator iterator){
        this.iter = iterator;
    }

    @Override
    public double calculateScore(T network) {
        U eval = newEval();

        if(network instanceof MultiLayerNetwork){
            DataSetIterator i = (iter != null ? iter : new MultiDataSetWrapperIterator(iterator));
            eval = ((MultiLayerNetwork) network).doEvaluation(i, eval)[0];
        } else if(network instanceof ComputationGraph){
            MultiDataSetIterator i = (iterator != null ? iterator : new MultiDataSetIteratorAdapter(iter));
            eval = ((ComputationGraph) network).doEvaluation(i, eval)[0];
        } else {
            throw new RuntimeException("Unknown model type: " + network.getClass());
        }
        return finalScore(eval);
    }

    protected abstract U newEval();

    protected abstract double finalScore(U eval);


}
