package org.deeplearning4j.caffe.translate;

import org.deeplearning4j.caffe.CaffeTestUtil;
import org.deeplearning4j.caffe.common.NNCofigBuilderContainer;
import org.deeplearning4j.caffe.common.SolverNetContainer;
import org.deeplearning4j.caffe.proto.Caffe;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author jeffreytang
 */
public class CaffeNetTranslatorTest {
    @Test
    public void testSolverLogisticTranslator() throws Exception {
        // Get SolverParamter
        SolverNetContainer container = CaffeTestUtil.getLogisticSolverNet();
        CaffeNetTranslator netTranslator = new CaffeNetTranslator();
        CaffeSolverTranslator solverTranslator = new CaffeSolverTranslator();

        Caffe.SolverParameter solver = container.getSolver();
        Caffe.NetParameter net = container.getNet();

        NNCofigBuilderContainer nnCofigBuilderContainer = new NNCofigBuilderContainer();

        // Parse SolverParameter and return wrapper container with solver parsed
        solverTranslator.translate(solver, nnCofigBuilderContainer);
        // Parse NetParameter and return wrapper container with solver parsed
        netTranslator.translate(net, nnCofigBuilderContainer);

        MultiLayerConfiguration model = nnCofigBuilderContainer.getListBuilder().build();

        assertTrue(model != null);
    }

    @Test
    public void testSolverImageNetTranslator() throws Exception {
        // Get SolverParamter
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

        MultiLayerConfiguration model = nnCofigBuilderContainer.getListBuilder().build();

        assertTrue(model != null);
    }

}
