package org.deeplearning4j.spark.impl.multilayer;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

public class ScoreFlatMapFunction implements FlatMapFunction<Iterator<DataSet>, Double> {

    private String json;
    private Broadcast<INDArray> params;
    private static Logger log = LoggerFactory.getLogger(IterativeReduceFlatMap.class);

    public ScoreFlatMapFunction(String json, Broadcast<INDArray> params){
        this.json = json;
        this.params = params;
    }

    @Override
    public Iterable<Double> call(Iterator<DataSet> dataSetIterator) throws Exception {
        if(!dataSetIterator.hasNext()) {
            return Collections.singletonList(0.0);
        }
        List<DataSet> collect = new ArrayList<>();
        while(dataSetIterator.hasNext()) {
            collect.add(dataSetIterator.next());
        }

        DataSet data = DataSet.merge(collect,false);
        if(log.isDebugEnabled()) {
            log.debug("Scoring {} examples with data {}",data.numExamples(), data.labelCounts());
        }

        MultiLayerNetwork network = new MultiLayerNetwork(MultiLayerConfiguration.fromJson(json));
        network.init();
        INDArray val = params.value();  //.value() object will be shared by all executors on each machine -> OK, as params are not modified by score function
        if(val.length() != network.numParams(false))
            throw new IllegalStateException("Network did not have same number of parameters as the broadcasted set parameters");
        network.setParameters(val);

        double score = network.score(data,false);
        if(network.conf().isMiniBatch()) score *= data.getFeatureMatrix().size(0);
        return Collections.singletonList(score);
    }
}
