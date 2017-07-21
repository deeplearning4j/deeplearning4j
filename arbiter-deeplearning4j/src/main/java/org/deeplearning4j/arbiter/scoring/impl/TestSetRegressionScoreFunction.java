package org.deeplearning4j.arbiter.scoring.impl;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.scoring.RegressionValue;
import org.deeplearning4j.arbiter.scoring.util.ScoreUtil;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Score function for regression (including multi-label regression) for a MultiLayerNetwork on a test set
 *
 * @author Alex Black
 */
public class TestSetRegressionScoreFunction implements ScoreFunction {
    private final RegressionValue regressionValue;

    /**
     * @param regressionValue The type of evaluation to do: MSE, MAE, RMSE, etc
     */
    public TestSetRegressionScoreFunction(RegressionValue regressionValue) {
        this.regressionValue = regressionValue;
    }

    @Override
    public double score(Object model, DataProvider dataProvider,
                    Map<String, Object> dataParameters) {
        DataSetIterator testData = ScoreUtil.getIterator(dataProvider.testData(dataParameters));
//        return ScoreUtil.score(model, testData, regressionValue);
        return 0.0;
    }

    @Override
    public List<Class<?>> getSupportedModelTypes() {
        return Arrays.<Class<?>>asList(MultiLayerNetwork.class, ComputationGraph.class);
    }

    @Override
    public List<Class<?>> getSupportedDataTypes() {
        return Arrays.<Class<?>>asList(DataSetIterator.class, MultiDataSetIterator.class);
    }

    @Override
    public boolean minimize() {
        return regressionValue != RegressionValue.CorrCoeff; //Maximize correlation coefficient, minimize the remaining ones
    }

    @Override
    public String toString() {
        return "TestSetRegressionScoreFunction(type=" + regressionValue + ")";
    }
}
