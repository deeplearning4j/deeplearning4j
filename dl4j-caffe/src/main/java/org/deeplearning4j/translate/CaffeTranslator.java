package org.deeplearning4j.translate;

import org.deeplearning4j.caffe.Caffe.*;

import java.io.*;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author jeffreytang
 */
public class CaffeTranslator {

    protected static Logger log = LoggerFactory.getLogger(CaffeTranslator.class);

    protected static void translateSolverNet(SolverNetBuilderContainer solverNetBuilder) {
        // logic to translate solver
        SolverParameter solver = solverNetBuilder.getSolver();
//        solver.
        // Instantiate builder and set global
        // setBuilder()

        // Must parse solver before parsing net, since solver contains global variables that are applied
        // across layers and after .layer() is called, Builder becomes ListBuilder, which can't set globals
        if (!solverNetBuilder.getParsedSolver())
            throw new UnsupportedOperationException("Have to parse SolverParameters before parsing NetParameter");

        // logic to translate net
        // CODE HERE
        // Instantiate ListBuilder and make layers
        // setListBuilder()
    }

    protected static MultiLayerConfiguration buildNNConfig(SolverNetBuilderContainer solverNetBuilder) {
        // Check if both net and solver are parsed
        if (!solverNetBuilder.getParsedState()) {
            throw new UnsupportedOperationException("Must call translateSolverNet before buildNNConfig");
        }

        //Build Builder to get NeuralNetConfiguration
        return solverNetBuilder.getListBuilder().build();
    }

    public static MultiLayerNetwork translate(String netPath, String solverPath,
                                                   boolean binaryFile) throws IOException {

        // Read the Caffe objects into a Java class
        SolverNetBuilderContainer solverNetContainer = CaffeLoader.load(netPath, solverPath, binaryFile);

        // Translate SolverParameter and NetParameter
        translateSolverNet(solverNetContainer);

        // Build MultiLayerConfiguration
        MultiLayerConfiguration conf = buildNNConfig(solverNetContainer);


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

}
