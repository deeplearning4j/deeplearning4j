package org.deeplearning4j.nn;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * Created by agibsonccc on 9/8/14.
 */
public abstract class BaseConvolutionalMultiLayerNetwork {
    private NeuralNetConfiguration conf;
    private Layer[] layers;
    private List<NeuralNetConfiguration> layerWiseConfigurations;
    private INDArray input;


    public void init() {

    }


}
