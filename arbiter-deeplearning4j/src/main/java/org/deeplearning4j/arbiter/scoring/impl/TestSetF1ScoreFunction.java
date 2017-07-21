package org.deeplearning4j.arbiter.scoring.impl;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.scoring.util.ScoreUtil;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Score function that calculates the F1 score
 * on a test set for a {@link MultiLayerNetwork}
 *
 * @author Alex Black
 */
public class TestSetF1ScoreFunction implements ScoreFunction {
    @Override
    public double score(Object model, DataProvider dataProvider, Map<String, Object> dataParameters) {
        DataSetIterator testData = ScoreUtil.getIterator(dataProvider.testData(dataParameters));
//        Evaluation evaluation = model.evaluate(testData);
//        return evaluation.f1();
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
        return false; //false -> maximize
    }

    @Override
    public String toString() {
        return "TestSetF1ScoreFunction";
    }
}
