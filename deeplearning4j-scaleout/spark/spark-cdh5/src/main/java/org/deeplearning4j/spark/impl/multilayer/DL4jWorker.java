package org.deeplearning4j.spark.impl.multilayer;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

/**
 * This is considered the "Worker"
 * This is the code that will run the .fit() method on the network
 *
 * the issue here is that this is getting called 1x per record
 * and before we could call it in a more controlled mini-batch setting
 *
 * @author josh
 * @author Adam Gibson
 */
public class DL4jWorker implements Function<DataSet, INDArray> {

    private final MultiLayerNetwork network;

    public DL4jWorker() {
        this.network = null;
    }

    public DL4jWorker(MultiLayerNetwork network) {
        this.network = network;
    }

    @Override
    public INDArray call(DataSet v1) throws Exception {
        network.fit(v1);
        System.out.println("DL4JWorker > call " + v1.numExamples() );
        return network.params();

    }
}