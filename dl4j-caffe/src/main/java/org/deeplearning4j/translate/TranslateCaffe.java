package org.deeplearning4j.translate;

import com.google.protobuf.TextFormat;
import org.deeplearning4j.caffe.Caffe.*;
import com.google.protobuf.CodedInputStream;

import java.io.*;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author jeffreytang
 */
public class TranslateCaffe {

    protected static Logger log = LoggerFactory.getLogger(TranslateCaffe.class);


    /**
     *
     * @param is InputStream of the caffemodel
     * @param sizeLimitMb Size limit of the CodedInputStream
     * @return NetParameter Java Class
     * @throws IOException
     */
    protected static NetParameter readBinaryNet(InputStream is, int sizeLimitMb) throws IOException {
        CodedInputStream codeStream = CodedInputStream.newInstance(is);
        // Increase the limit when loading bigger caffemodels size
        codeStream.setSizeLimit(sizeLimitMb * 1024 * 1024);
        return NetParameter.parseFrom(codeStream);
    }

    /**
     *
     * @param binaryNetPath Path of caffemodel
     * @param sizeLimitMb Size limit of the CodedInputStream
     * @return NetParameter Java Class
     * @throws IOException
     */
    protected static NetParameter readBinaryNet(String binaryNetPath, int sizeLimitMb) throws IOException {
        InputStream is = new BufferedInputStream(new FileInputStream(binaryNetPath));
        return readBinaryNet(is, sizeLimitMb);
    }

    protected static NetParameter readTextFormatNet(String textFormatNetPath) throws IOException{

        InputStream is = new FileInputStream(textFormatNetPath);
        InputStreamReader isReader = new InputStreamReader(is, "ASCII");

        NetParameter.Builder builder = NetParameter.newBuilder();
        TextFormat.merge(isReader, builder);
        return builder.build();
    }

    protected static SolverParameter readTextFormatSolver(String textFormatSolverPath) throws IOException {

        InputStream is = new FileInputStream(textFormatSolverPath);
        InputStreamReader isReader = new InputStreamReader(is, "ASCII");

        SolverParameter.Builder builder = SolverParameter.newBuilder();
        TextFormat.merge(isReader, builder);
        return builder.build();
    }

    protected static SolverNetBuilderContainer read(String netPath,
                                                String solverPath,
                                                boolean binaryFile) throws IOException {
        NetParameter net;
        SolverParameter solver;

        if (binaryFile) {
            log.info("Reading in binary format Caffe netowrk configurations");
            try {
                net = readBinaryNet(netPath, 10000);
            } catch (OutOfMemoryError e) {
                throw new OutOfMemoryError("Model is bigger than 10GB. If you want o raise the limit, " +
                        "specify the sizeLimitMb");
            }
        } else {
            net = readTextFormatNet(netPath);
        }

        solver = readTextFormatSolver(solverPath);

        return new SolverNetBuilderContainer(solver, net);
    }

    protected static void translateSolverNet(SolverNetBuilderContainer solverNetBuilder) {
        // logic to translate solver
        // CODE HERE
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
        SolverNetBuilderContainer solverNetContainer = read(netPath, solverPath, binaryFile);

        // Translate SolverParameter and NetParameter
        translateSolverNet(solverNetContainer);

        // Build MultiLayerConfiguration
        MultiLayerConfiguration conf = buildNNConfig(solverNetContainer);


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

}
