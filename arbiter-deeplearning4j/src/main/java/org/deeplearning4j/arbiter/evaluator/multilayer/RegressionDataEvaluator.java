package org.deeplearning4j.arbiter.evaluator.multilayer;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.evaluation.ModelEvaluator;
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
 * Created by agibsonccc on 3/12/17.
 */
@AllArgsConstructor
public class RegressionDataEvaluator implements ModelEvaluator {
    private RegressionValue regressionValue;
    private Map<String, Object> params = null;

    @Override
    public Double evaluateModel(Object model, DataProvider dataProvider) {

        if (model instanceof MultiLayerNetwork) {
            DataSetIterator iterator = ScoreUtil.getIterator(dataProvider.testData(params));
            return ScoreUtil.score((MultiLayerNetwork) model, iterator, regressionValue);
        } else {
            DataSetIterator iterator = ScoreUtil.getIterator(dataProvider.testData(params));
            return ScoreUtil.score((ComputationGraph) model, iterator, regressionValue);
        }
    }

    @Override
    public List<Class<?>> getSupportedModelTypes() {
        return Arrays.<Class<?>>asList(MultiLayerNetwork.class, ComputationGraph.class);
    }

    @Override
    public List<Class<?>> getSupportedDataTypes() {
        return Arrays.<Class<?>>asList(DataSetIterator.class, MultiDataSetIterator.class);
    }
}
