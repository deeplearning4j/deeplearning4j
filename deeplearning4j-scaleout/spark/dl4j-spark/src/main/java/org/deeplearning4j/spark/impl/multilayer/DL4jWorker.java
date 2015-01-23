package org.deeplearning4j.spark.impl.multilayer;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This is considered the "Worker"
 * This is the code that will run the .fitDataSet() method on the network
 *
 * the issue here is that this is getting called 1x per record
 * and before we could call it in a more controlled mini-batch setting
 *
 * @author josh
 * @author Adam Gibson
 */
public class DL4jWorker implements Function<DataSet, INDArray> {

    private final Model network;
    private static Logger log = LoggerFactory.getLogger(DL4jWorker.class);
    public DL4jWorker(String json,INDArray params) {
        this.network = new MultiLayerNetwork(MultiLayerConfiguration.fromJson(json));
        MultiLayerNetwork thisNetwork = (MultiLayerNetwork) this.network;
        thisNetwork.init();
        int numParams = this.network.numParams();
        if(numParams != params.length())
            throw new IllegalStateException("Number of params for configured network was " + numParams + " while the specified parameter vector length was " + params.length());
        thisNetwork.setParameters(params);
    }

    @Override
    public INDArray call(DataSet v1) throws Exception {
        MultiLayerNetwork thisNetwork = (MultiLayerNetwork) this.network;
        thisNetwork.fit(v1);
        return network.params();

    }
}