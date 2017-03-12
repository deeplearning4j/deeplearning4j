package org.deeplearning4j.arbiter.scoring.multilayer;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.scoring.RegressionValue;
import org.deeplearning4j.arbiter.scoring.util.ScoreUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Map;

/**
 * Score function for regression (including multi-label regression) for a MultiLayerNetwork on a test set
 *
 * @author Alex Black
 */
public class TestSetRegressionScoreFunction implements ScoreFunction<MultiLayerNetwork, DataSetIterator> {
    private final RegressionValue regressionValue;

    /**
     * @param regressionValue The type of evaluation to do: MSE, MAE, RMSE, etc
     */
    public TestSetRegressionScoreFunction(RegressionValue regressionValue) {
        this.regressionValue = regressionValue;
    }

    @Override
    public double score(MultiLayerNetwork model, DataProvider<DataSetIterator> dataProvider, Map<String, Object> dataParameters) {
        DataSetIterator testSet = dataProvider.testData(dataParameters);
        return ScoreUtil.score(model,testSet,regressionValue);
    }

    @Override
    public boolean minimize() {
        return regressionValue != RegressionValue.CorrCoeff;    //Maximize correlation coefficient, minimize the remaining ones
    }

    @Override
    public String toString() {
        return "TestSetRegressionScoreFunction(type=" + regressionValue + ")";
    }
}
