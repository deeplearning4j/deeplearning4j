package org.deeplearning4j.spark.impl.multilayer;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.berkeley.Pair;
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
public class DL4jWorker implements Function<Pair<MultiLayerNetwork,DataSet>, INDArray> {

    private static Logger log = LoggerFactory.getLogger(DL4jWorker.class);


    @Override
    public  INDArray call(Pair<MultiLayerNetwork,DataSet> v1) throws Exception {
        MultiLayerNetwork network = v1.getFirst();
        network.initialize(v1.getSecond());
        network.fit(v1.getSecond());
        INDArray params = network.params();
        return params;
    }

}