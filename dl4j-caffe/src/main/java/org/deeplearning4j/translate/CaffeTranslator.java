package org.deeplearning4j.translate;

import org.deeplearning4j.caffe.Caffe.NetParameter;
import org.deeplearning4j.caffe.Caffe.SolverParameter;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
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
            throw new UnsupportedOperationException("SolverNetContainer must container SolverParameter (solver)" +
                    " and NetParameter (net) ");
        }
    }

    protected MultiLayerConfiguration translate() {

        // Parse SolverParameter and return wrapper container with solver parsed
        nnCofigBuilderContainer = CaffeSolverTranslator.translate(solver, nnCofigBuilderContainer);
        // Parse NetParameter and return wrapper container with solver parsed
        nnCofigBuilderContainer = CaffeNetTranslator.translate(net, nnCofigBuilderContainer);
        // Get ListBuilder and build
        return nnCofigBuilderContainer.getListBuilder().build();
    }
}
