package org.deeplearning4j.arbiter.scoring.graph;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.scoring.RegressionValue;
import org.deeplearning4j.arbiter.scoring.util.ScoreUtil;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.Map;

/**
 * Score function for regression (including multi-label regression, and multiple output arrays) for a ComputationGraph
 * on a MultiDataSetIterator
 *
 * @author Alex Black
 */
public class GraphTestSetRegressionScoreFunction implements ScoreFunction<ComputationGraph, Object> {

    private final RegressionValue regressionValue;

    /**
     * @param regressionValue The type of evaluation to do: MSE, MAE, RMSE, etc
     */
    public GraphTestSetRegressionScoreFunction(RegressionValue regressionValue) {
        this.regressionValue = regressionValue;
    }

    @Override
    public double score(ComputationGraph model, DataProvider<Object> dataProvider, Map<String, Object> dataParameters) {
        MultiDataSetIterator testSet = ScoreUtil.getMultiIterator(dataProvider.testData(dataParameters));
        return ScoreUtil.score(model,testSet,regressionValue);
    }

    @Override
    public boolean minimize() {
        return regressionValue != RegressionValue.CorrCoeff;    //Maximize correlation coefficient, minimize the remaining ones
    }

    @Override
    public String toString() {
        return "GraphTestSetRegressionScoreFunction(type=" + regressionValue + ")";
    }
}
