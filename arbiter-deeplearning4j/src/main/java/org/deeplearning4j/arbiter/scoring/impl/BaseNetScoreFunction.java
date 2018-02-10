package org.deeplearning4j.arbiter.scoring.impl;

import lombok.EqualsAndHashCode;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Created by Alex on 23/07/2017.
 */
@EqualsAndHashCode
public abstract class BaseNetScoreFunction implements ScoreFunction {


    @Override
    public double score(Object model, DataProvider dataProvider, Map<String, Object> dataParameters) {
        Object testData = dataProvider.testData(dataParameters);
        if (model instanceof MultiLayerNetwork) {
            if (testData instanceof DataSetIterator) {
                return score((MultiLayerNetwork) model, (DataSetIterator) testData);
            } else if(testData instanceof MultiDataSetIterator){
                return score((MultiLayerNetwork) model, (MultiDataSetIterator) testData);
            } else if(testData instanceof DataSetIteratorFactory){
                return score((MultiLayerNetwork)model, ((DataSetIteratorFactory)testData).create());
            } else {
                throw new RuntimeException("Unknown type of data: " + testData.getClass());
            }
        } else {
            if (testData instanceof DataSetIterator) {
                return score((ComputationGraph) model, (DataSetIterator) testData);
            } else if(testData instanceof DataSetIteratorFactory){
                return score((ComputationGraph) model, ((DataSetIteratorFactory)testData).create());
            } else if(testData instanceof MultiDataSetIterator) {
                return score((ComputationGraph) model, (MultiDataSetIterator) testData);
            } else {
                throw new RuntimeException("Unknown type of data: " + testData.getClass());
            }
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

    public abstract double score(MultiLayerNetwork net, DataSetIterator iterator);

    public abstract double score(MultiLayerNetwork net, MultiDataSetIterator iterator);

    public abstract double score(ComputationGraph graph, DataSetIterator iterator);

    public abstract double score(ComputationGraph graph, MultiDataSetIterator iterator);
}
