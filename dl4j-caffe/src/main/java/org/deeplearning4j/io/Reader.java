package io;

import com.fasterxml.jackson.dataformat.protobuf.schema.ProtobufSchema;
import com.fasterxml.jackson.dataformat.protobuf.schema.ProtobufSchemaLoader;
import org.springframework.core.io.ClassPathResource;

import java.io.*;
import java.nio.charset.Charset;

import com.google.common.io.Files;

/**
 * Created by jeffreytang on 7/4/15.
 */
public class Reader {


    public static ProtobufSchema getSchema(String filePath, Boolean resource) throws IOException {
        // Get the resulting file depending if the file is located in the Resource directory
        File finalFile;
        if (resource) {
            finalFile = new ClassPathResource(filePath).getFile();
        } else {
            finalFile = new File(filePath);
        }

        // Convert file to ProtobufSchema
        String caffeProtoString = Files.toString(finalFile, Charset.defaultCharset());
        return ProtobufSchemaLoader.std.parse(caffeProtoString);
    }

    public static ProtobufSchema getCaffeSchema() throws IOException {
        return getSchema("proto/caffe.proto", true);
    }

}