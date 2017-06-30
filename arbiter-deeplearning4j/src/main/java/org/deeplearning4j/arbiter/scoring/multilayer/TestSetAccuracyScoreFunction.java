package org.deeplearning4j.arbiter.scoring.multilayer;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.scoring.util.ScoreUtil;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Map;

/**
 * Score function that calculates the accuracy on a
 * test set for a {@link MultiLayerNetwork}
 *
 * @author Alex Black
 */
public class TestSetAccuracyScoreFunction implements ScoreFunction<MultiLayerNetwork, Object> {
    @Override
    public double score(MultiLayerNetwork model, DataProvider<Object> dataProvider, Map<String, Object> dataParameters) {
        DataSetIterator testData = ScoreUtil.getIterator(dataProvider.testData(dataParameters));
        Evaluation evaluation = model.evaluate(testData);
        return evaluation.accuracy();
    }

    @Override
    public boolean minimize() {
        return false;   //false -> maximize
    }

    @Override
    public String toString() {
        return "TestSetAccuracyScoreFunction";
    }
}
