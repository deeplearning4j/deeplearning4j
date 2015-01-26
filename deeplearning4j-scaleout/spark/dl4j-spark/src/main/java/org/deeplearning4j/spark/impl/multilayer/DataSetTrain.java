package org.deeplearning4j.spark.impl.multilayer;

import org.apache.spark.Accumulator;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 1/25/15.
 */
public class DataSetTrain implements VoidFunction<DataSet> {
    private Accumulator<INDArray> accum;
    private String json;
    private Broadcast<INDArray> params;
    private static Logger log = LoggerFactory.getLogger(DataSetTrain.class);

    public DataSetTrain(Accumulator<INDArray> accum, Broadcast<INDArray> params, String json) {
        this.accum = accum;
        this.params = params;
        this.json = json;
    }

    @Override
    public void call(DataSet dataSet) throws Exception {
        MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(json);
        if(dataSet.numExamples() != conf.getConf(0).getBatchSize())
            return;

        conf = MultiLayerConfiguration.fromJson(json);
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setParameters(params.value());
        network.fit(dataSet);
        accum.add(network.params());
    }



}
