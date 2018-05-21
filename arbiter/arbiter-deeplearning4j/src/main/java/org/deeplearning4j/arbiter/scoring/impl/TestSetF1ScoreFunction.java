package org.deeplearning4j.arbiter.scoring.impl;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Score function that calculates the F1 score
 * on a test set for a {@link MultiLayerNetwork} or {@link ComputationGraph}
 *
 * @author Alex Black
 * @deprecated Use {@link EvaluationScoreFunction}
 */
@Data
@EqualsAndHashCode(callSuper = true)
@Deprecated
public class TestSetF1ScoreFunction extends BaseNetScoreFunction {

    @Override
    public boolean minimize() {
        return false; //false -> maximize
    }


    @Override
    public String toString() {
        return "TestSetF1ScoreFunction";
    }

    @Override
    public double score(MultiLayerNetwork net, DataSetIterator iterator) {
        Evaluation e = net.evaluate(iterator);
        return e.f1();
    }

    @Override
    public double score(MultiLayerNetwork net, MultiDataSetIterator iterator) {
        throw new UnsupportedOperationException("Cannot evaluate MultiLayerNetwork on MultiDataSetIterator");
    }

    @Override
    public double score(ComputationGraph graph, DataSetIterator iterator) {
        Evaluation e = graph.evaluate(iterator);
        return e.f1();
    }

    @Override
    public double score(ComputationGraph graph, MultiDataSetIterator iterator) {
        Evaluation e = graph.evaluate(iterator);
        return e.f1();
    }
}
