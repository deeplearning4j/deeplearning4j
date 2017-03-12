package org.deeplearning4j.arbiter.scoring.multilayer;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Map;

/**
 * Score function that calculates the F1 score on a test set for a MultiLayerNetwork
 *
 * @author Alex Black
 */
public class TestSetF1ScoreFunction implements ScoreFunction<MultiLayerNetwork, DataSetIterator> {
    @Override
    public double score(MultiLayerNetwork model, DataProvider<DataSetIterator> dataProvider, Map<String, Object> dataParameters) {
        DataSetIterator testData = dataProvider.testData(dataParameters);
        Evaluation evaluation = model.evaluate(testData);
        return evaluation.f1();
    }

    @Override
    public boolean minimize() {
        return false;   //false -> maximize
    }

    @Override
    public String toString() {
        return "TestSetF1ScoreFunction";
    }
}
