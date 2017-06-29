package org.deeplearning4j.arbiter.evaluator.multilayer;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.evaluation.ModelEvaluator;
import org.deeplearning4j.arbiter.scoring.RegressionValue;
import org.deeplearning4j.arbiter.scoring.util.ScoreUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.Map;

/**
 * Created by agibsonccc on 3/12/17.
 */
@AllArgsConstructor
public class RegressionDataSetEvaluator implements ModelEvaluator<MultiLayerNetwork, Object, Double> {
    private RegressionValue regressionValue;
    private Map<String,Object> evalParameters = null;

    @Override
    public Double evaluateModel(MultiLayerNetwork model, DataProvider<Object> dataProvider) {
        return ScoreUtil.score(model,ScoreUtil.getIterator(dataProvider.testData(evalParameters)),regressionValue);
    }
}
