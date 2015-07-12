package org.deeplearning4j.translate;

import org.deeplearning4j.caffe.Caffe.*;
import com.google.protobuf.CodedInputStream;
import org.springframework.core.io.ClassPathResource;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Created by jeffreytang on 7/9/15.
 */
public class CaffeModelToJavaClass {

    public static NetParameter readCaffeModel(String caffeModelPath, int sizeLimitMb) throws IOException {
        InputStream is = new FileInputStream(caffeModelPath);
        CodedInputStream codeStream = CodedInputStream.newInstance(is);
        // Increase the limit when loading bigger caffemodels size
        int oldLimit = codeStream.setSizeLimit(sizeLimitMb * 1024 * 1024);
        return NetParameter.parseFrom(codeStream);
    }
}
