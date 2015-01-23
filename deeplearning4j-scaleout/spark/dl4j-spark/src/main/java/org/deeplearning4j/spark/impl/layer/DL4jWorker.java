package org.deeplearning4j.spark.impl.layer;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

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

    public DL4jWorker(String json,INDArray params) {
        NeuralNetConfiguration conf = NeuralNetConfiguration.fromJson(json);
        LayerFactory layerFactory = conf.getLayerFactory();
        if(layerFactory == null)
            throw new IllegalStateException("Please specify a layer factory");
        this.network = layerFactory.create(conf);
        int numParams = this.network.numParams();
        if(numParams != params.length())
            throw new IllegalStateException("Number of params for configured network was " + numParams + " while the specified parameter vector length was " + params.length());
        Layer network = (Layer) this.network;
        network.setParameters(params);
    }

    @Override
    public INDArray call(DataSet v1) throws Exception {
        Layer network = (Layer) this.network;
        network.fit(v1.getFeatureMatrix());
        return network.params();

    }
}