package org.deeplearning4j.caffe.create;

import org.deeplearning4j.caffe.CaffeTestUtil;
import org.deeplearning4j.caffe.common.NNCofigBuilderContainer;
import org.deeplearning4j.caffe.common.SolverNetContainer;
import org.deeplearning4j.caffe.dag.CaffeNode;
import org.deeplearning4j.caffe.dag.Graph;
import org.deeplearning4j.caffe.proto.Caffe;
import org.deeplearning4j.caffe.translate.CaffeLayerGraphConversion;
import org.deeplearning4j.caffe.translate.CaffeNetTranslator;
import org.deeplearning4j.caffe.translate.CaffeSolverTranslator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;

/**
 * @author Adam Gibson
 */
public class Dl4jCaffeNetCreatorTest {
    @Test
    public void testCreateNet() throws Exception {
        SolverNetContainer container = CaffeTestUtil.getImageNetSolverNet();
        CaffeNetTranslator netTranslator = new CaffeNetTranslator();
        CaffeSolverTranslator solverTranslator = new CaffeSolverTranslator();

        Caffe.SolverParameter solver = container.getSolver();
        Caffe.NetParameter net = container.getNet();

        NNCofigBuilderContainer nnCofigBuilderContainer = new NNCofigBuilderContainer();

        // Parse SolverParameter and return wrapper container with solver parsed
        solverTranslator.translate(solver, nnCofigBuilderContainer);
        // Parse NetParameter and return wrapper container with solver parsed
        netTranslator.translate(net, nnCofigBuilderContainer);


        Graph<CaffeNode> graph = new CaffeLayerGraphConversion(net).convert();
        DL4jCaffeNetCreator creator = new DL4jCaffeNetCreator();
        MultiLayerNetwork network = creator.createNet(nnCofigBuilderContainer, graph);
    }

}
