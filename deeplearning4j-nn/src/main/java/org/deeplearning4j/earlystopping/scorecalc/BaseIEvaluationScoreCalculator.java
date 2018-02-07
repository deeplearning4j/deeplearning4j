package org.deeplearning4j.earlystopping.scorecalc;

import lombok.AllArgsConstructor;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

@AllArgsConstructor
public abstract class BaseIEvaluationScoreCalculator<T extends Model, U extends IEvaluation> implements ScoreCalculator<T> {

    protected DataSetIterator iterator;

    @Override
    public double calculateScore(T network) {
        U eval = newEval();

        if(network instanceof MultiLayerNetwork){
            eval = ((MultiLayerNetwork) network).doEvaluation(iterator, eval)[0];
        } else if(network instanceof ComputationGraph){
            eval = ((ComputationGraph) network).doEvaluation(iterator, eval)[0];
        } else {
            throw new RuntimeException("Unknown model type: " + network.getClass());
        }
        return finalScore(eval);
    }

    protected abstract U newEval();

    protected abstract double finalScore(U eval);


}
