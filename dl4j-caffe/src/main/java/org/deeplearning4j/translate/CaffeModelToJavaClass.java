package org.deeplearning4j.translate;

import org.deeplearning4j.caffe.Caffe.*;
import com.google.protobuf.CodedInputStream;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

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
    public static NetParameter readCaffeModel(InputStream is, int sizeLimitMb) throws IOException {
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
    public static NetParameter readCaffeModel(String caffeModelPath, int sizeLimitMb) throws IOException {
        InputStream is = new BufferedInputStream(new FileInputStream(caffeModelPath));
        return readCaffeModel(is,sizeLimitMb);
    }
}
