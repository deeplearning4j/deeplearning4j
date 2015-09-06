package org.deeplearning4j.caffe.create;

import org.deeplearning4j.caffe.common.NNCofigBuilderContainer;
import org.deeplearning4j.caffe.dag.CaffeNode;
import org.deeplearning4j.caffe.dag.Graph;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class DL4jCaffeNetCreator {
    /**
     * Creates a network
     * @param nnCofigBuilderContainer
     * @param graph
     * @return
     */
    public MultiLayerNetwork createNet(NNCofigBuilderContainer nnCofigBuilderContainer,Graph<CaffeNode> graph) {
        MultiLayerConfiguration conf = nnCofigBuilderContainer.getListBuilder().build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        List<INDArray> arrs = new ArrayList<>();
        for(CaffeNode node : graph.getAllNodes()) {
            for(INDArray arr : node.getData())
                arrs.add(arr.ravel());
        }

        network.setParameters(Nd4j.toFlattened(arrs));
        return network;
    }

}
