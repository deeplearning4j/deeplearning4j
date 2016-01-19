package org.deeplearning4j.spark.impl.computationgraph;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

public class ScoreFlatMapFunctionCGMultiDataSet implements FlatMapFunction<Iterator<MultiDataSet>, Double> {

    private String json;
    private Broadcast<INDArray> params;
    private static Logger log = LoggerFactory.getLogger(IterativeReduceFlatMapCG.class);

    public ScoreFlatMapFunctionCGMultiDataSet(String json, Broadcast<INDArray> params){
        this.json = json;
        this.params = params;
    }

    @Override
    public Iterable<Double> call(Iterator<MultiDataSet> dataSetIterator) throws Exception {
        if(!dataSetIterator.hasNext()) {
            return Collections.singletonList(0.0);
        }
        List<MultiDataSet> collect = new ArrayList<>();
        while(dataSetIterator.hasNext()) {
            collect.add(dataSetIterator.next());
        }

        MultiDataSet data = org.nd4j.linalg.dataset.MultiDataSet.merge(collect);

        ComputationGraph network = new ComputationGraph(ComputationGraphConfiguration.fromJson(json));
        network.init();
        INDArray val = params.value();
        if(val.length() != network.numParams(false))
            throw new IllegalStateException("Network did not have same number of parameters as the broadcast set parameters");
        network.setParams(val);

        double score = network.score(data,false);
        if(network.conf().isMiniBatch()) score *= data.getFeatures(0).size(0);
        return Collections.singletonList(score);
    }
}
