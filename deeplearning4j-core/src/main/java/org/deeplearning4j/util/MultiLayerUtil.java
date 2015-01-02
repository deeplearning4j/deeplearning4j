package org.deeplearning4j.util;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Various functions for manipulating a multi layer network
 * @author Adam Gibson
 */
public class MultiLayerUtil {
    /**
     * Return the weight matrices for a multi layer network
     * @param network the network to get the weights for
     * @return the weight matrices for a given multi layer network
     */
    public static List<INDArray> weightMatrices(MultiLayerNetwork network) {
        List<INDArray> ret = new ArrayList<>();
        for(int i = 0; i < network.getLayers().length; i++) {
            ret.add(network.getLayers()[i].getParam(DefaultParamInitializer.WEIGHT_KEY));
        }
        return ret;
    }


}
