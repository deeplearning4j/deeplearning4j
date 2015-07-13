package org.deeplearning4j.translate;

import com.google.protobuf.TextFormat;
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
    public static NetParameter readBinaryCaffeModel(InputStream is, int sizeLimitMb) throws IOException {
        CodedInputStream codeStream = CodedInputStream.newInstance(is);
        // Increase the limit when loading bigger caffemodels size
        int oldLimit = codeStream.setSizeLimit(sizeLimitMb * 1024 * 1024);
        return NetParameter.parseFrom(codeStream);
    }

    /**
     *
     * @param caffeModelPath Path of caffemodel
     * @param sizeLimitMb Size limit of the CodedInputStream
     * @return NetParameter Java Class
     * @throws IOException
     */
    public static NetParameter readBinaryCaffeModel(String caffeModelPath, int sizeLimitMb) throws IOException {
        InputStream is = new BufferedInputStream(new FileInputStream(caffeModelPath));
        return readBinaryCaffeModel(is, sizeLimitMb);
    }

    public static NetParameter readTextFormatNetProto(String caffeNetProtoPath) throws IOException{

        InputStream is = new FileInputStream(caffeNetProtoPath);
        InputStreamReader isReader = new InputStreamReader(is, "ASCII");

        NetParameter.Builder builder = NetParameter.newBuilder();
        TextFormat.merge(isReader, builder);
        return builder.build();
    }

    public static SolverParameter readTextFormatSolverProto(String caffeSolverProtoPath) throws IOException {

        InputStream is = new FileInputStream(caffeSolverProtoPath);
        InputStreamReader isReader = new InputStreamReader(is, "ASCII");

        SolverParameter.Builder builder = SolverParameter.newBuilder();
        TextFormat.merge(isReader, builder);
        return builder.build();
    }

//    public static NetParameter readCaffeWithWeights() {}
//
//    public static NetParameter readCaffeWithoutWeights() {}

}
