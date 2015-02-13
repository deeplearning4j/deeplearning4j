package org.deeplearning4j.spark.impl.multilayer;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Runs iterative reduce on the dataset
 * @author Adam Gibson
 */
public class IterativeReduce implements Function<DataSet,INDArray> {
    private String json;
    private Broadcast<INDArray> params;
    private static final Logger log = LoggerFactory.getLogger(IterativeReduce.class);

    /**
     * Train and average over mini batches from a dataset
     * @param params the parameters that were broadcast
     * @param json the configuration for the network
     */
    public IterativeReduce(Broadcast<INDArray> params, String json) {
        this.params = params;
        this.json = json;
    }

    @Override
    public INDArray call(DataSet dataSet) throws Exception {
        log.info("Training on " + dataSet.numExamples());
        MultiLayerConfiguration conf =  MultiLayerConfiguration.fromJson(json);
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setParameters(params.value());
        network.fit(dataSet);
        return network.params();
    }



}
