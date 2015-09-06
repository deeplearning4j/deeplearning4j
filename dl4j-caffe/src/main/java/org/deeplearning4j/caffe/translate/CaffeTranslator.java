package org.deeplearning4j.caffe.translate;

import org.deeplearning4j.caffe.proto.Caffe.NetParameter;
import org.deeplearning4j.caffe.proto.Caffe.SolverParameter;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.caffe.common.NNCofigBuilderContainer;
import org.deeplearning4j.caffe.common.SolverNetContainer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author jeffreytang
 */
public class CaffeTranslator {

    private SolverParameter solver;
    private NetParameter net;
    private NNCofigBuilderContainer nnCofigBuilderContainer;

    protected static Logger log = LoggerFactory.getLogger(CaffeTranslator.class);

    public CaffeTranslator(SolverNetContainer solverNet) {
        if (solverNet.getSolver() != null && solverNet.getNet() != null) {
            this.solver = solverNet.getSolver();
            this.net = solverNet.getNet();
            this.nnCofigBuilderContainer = new NNCofigBuilderContainer();
        } else {
            throw new IllegalArgumentException("SolverNetContainer must container SolverParameter (solver)" +
                    " and NetParameter (net) ");
        }
    }

    protected MultiLayerConfiguration translate() throws Exception {

        CaffeNetTranslator netTranslator = new CaffeNetTranslator();
        CaffeSolverTranslator solverTranslator = new CaffeSolverTranslator();

        // Parse SolverParameter and return wrapper container with solver parsed
        solverTranslator.translate(solver, nnCofigBuilderContainer);
        // Parse NetParameter and return wrapper container with solver parsed
        netTranslator.translate(net, nnCofigBuilderContainer);
        // Get ListBuilder and build
        return nnCofigBuilderContainer.getListBuilder().build();
    }
}
