package org.deeplearning4j.translate;

import com.google.protobuf.TextFormat;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.caffe.Caffe.*;
import com.google.protobuf.CodedInputStream;

import java.io.*;

/**
 * @author jeffreytang
 */
public class CaffeModelToJavaClass {

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
        int oldLimit = codeStream.setSizeLimit(sizeLimitMb * 1024 * 1024);
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


    // Define the wrapper class and methods to return the wrapper class
    @AllArgsConstructor @Data
    public static class CaffeSolverNetContainer {
        NetParameter net;
        SolverParameter solver;
    }

    public static CaffeSolverNetContainer readCaffeWithWeights(String binaryNetPath,
                                                               String textFormatSolverPath,
                                                               int sizeLimitMb) throws IOException {
        NetParameter net = readBinaryNet(binaryNetPath, sizeLimitMb);
        SolverParameter solver = readTextFormatSolver(textFormatSolverPath);
        return new CaffeSolverNetContainer(net, solver);
    }

    public static CaffeSolverNetContainer readCaffeWithoutWeights(String textFormatNetPath,
                                                                  String textFormatSolverPath) throws IOException {
        NetParameter net = readTextFormatNet(textFormatNetPath);
        SolverParameter solver = readTextFormatSolver(textFormatSolverPath);
        return new CaffeSolverNetContainer(net, solver);
    }

}
