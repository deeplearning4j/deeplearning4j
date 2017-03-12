package org.deeplearning4j.arbiter.scoring.graph.factory;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.scoring.RegressionValue;
import org.deeplearning4j.arbiter.scoring.graph.util.ScoreUtil;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIteratorFactory;

import java.util.Map;

/**
 * Score function for regression (including multi-label regression, and multiple output arrays) for a ComputationGraph
 * on a MultiDataSetIteratorFactory
 *
 * @author Alex Black
 */
public class GraphTestSetRegressionScoreFunction implements ScoreFunction<ComputationGraph, MultiDataSetIteratorFactory> {

    private final RegressionValue regressionValue;

    /**
     * @param regressionValue The type of evaluation to do: MSE, MAE, RMSE, etc
     */
    public GraphTestSetRegressionScoreFunction(RegressionValue regressionValue) {
        this.regressionValue = regressionValue;
    }

    @Override
    public double score(ComputationGraph model, DataProvider<MultiDataSetIteratorFactory> dataProvider, Map<String, Object> dataParameters) {
        return ScoreUtil.score(model,dataProvider.testData(dataParameters).create(),regressionValue);
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
