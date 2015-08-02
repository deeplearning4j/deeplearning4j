package org.deeplearning4j.translate;

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.TextFormat;
import org.deeplearning4j.caffe.Caffe;

import java.io.*;

/**
 * @author jeffreytang
 */
public class ReadCaffe {

    /**
     *
     * @param is InputStream of the caffemodel
     * @param sizeLimitMb Size limit of the CodedInputStream
     * @return NetParameter Java Class
     * @throws IOException
     */
    protected static Caffe.NetParameter readBinaryNet(InputStream is, int sizeLimitMb) throws IOException {
        CodedInputStream codeStream = CodedInputStream.newInstance(is);
        // Increase the limit when loading bigger caffemodels size
        // Concern about having int as the size limit. There is a limit to that!
        codeStream.setSizeLimit(sizeLimitMb * 1024 * 1024);
        return Caffe.NetParameter.parseFrom(codeStream);
    }

    /**
     *
     * @param binaryNetPath Path of caffemodel
     * @param sizeLimitMb Size limit of the CodedInputStream
     * @return NetParameter Java Class
     * @throws IOException
     */
    protected static Caffe.NetParameter readBinaryNet(String binaryNetPath, int sizeLimitMb) throws IOException {
        InputStream is = new BufferedInputStream(new FileInputStream(binaryNetPath));
        return readBinaryNet(is, sizeLimitMb);
    }

    /**
     *
     * @param textFormatNetPath Path of the file that specifies NetParameter
     * @return NetParameter Java Class
     * @throws IOException
     */
    protected static Caffe.NetParameter readTextFormatNet(String textFormatNetPath) throws IOException{

        InputStream is = new FileInputStream(textFormatNetPath);
        InputStreamReader isReader = new InputStreamReader(is, "ASCII");

        Caffe.NetParameter.Builder builder = Caffe.NetParameter.newBuilder();
        TextFormat.merge(isReader, builder);
        return builder.build();
    }

    /**
     *
     * @param textFormatFile File object that specifies NetParameter
     * @return NetParameter Java Class
     * @throws IOException
     */
    protected static Caffe.NetParameter readTextFormatNet(File textFormatFile) throws IOException{
        return readTextFormatNet(textFormatFile.getPath());
    }

    /**
     *
     * @param textFormatSolverPath Path of the file that specifies SolverParameter
     * @return SolverParameter Java Class
     * @throws IOException
     */
    protected static Caffe.SolverParameter readTextFormatSolver(String textFormatSolverPath) throws IOException {

        try(InputStream is = new FileInputStream(textFormatSolverPath)) {
            InputStreamReader isReader = new InputStreamReader(is, "ASCII");

            Caffe.SolverParameter.Builder builder = Caffe.SolverParameter.newBuilder();
            TextFormat.merge(isReader, builder);

            return builder.build();
        }

    }

    /**
     *
     * @param textFormatSolverFile File object that specifies SolverParameter
     * @return SolverParameter Java Class
     * @throws IOException
     */
    protected static Caffe.SolverParameter readTextFormatSolver(File textFormatSolverFile) throws IOException {
        return readTextFormatSolver(textFormatSolverFile.getPath());
    }

}
