package org.deeplearning4j.caffe.create;

import org.deeplearning4j.caffe.common.NNConfigBuilderContainer;
import org.deeplearning4j.caffe.dag.CaffeNode;
import org.deeplearning4j.caffe.dag.Graph;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
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
     * @param nnConfigBuilderContainer
     * @param graph
     * @return
     */
    public MultiLayerNetwork createNet(NNConfigBuilderContainer nnConfigBuilderContainer,Graph<CaffeNode> graph) {
        MultiLayerConfiguration conf = nnConfigBuilderContainer.getListBuilder().build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        List<INDArray> arrs = new ArrayList<>();
        for(CaffeNode node : graph.getAllNodes()) {
            for(INDArray arr : node.getData())
                arrs.add(arr.ravel());
        }

        INDArray params = Nd4j.toFlattened(arrs);
        if(params.length() != network.numParams())
            throw new IllegalStateException("Params length must be equal to " + network.numParams() + " but was " + params.length());
        network.setParameters(Nd4j.toFlattened(arrs));
        return network;
    }

}
