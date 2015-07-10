package translate;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.protobuf.ProtobufFactory;
import com.fasterxml.jackson.dataformat.protobuf.schema.ProtobufSchema;
import org.springframework.core.io.ClassPathResource;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;


/**
 * Created by jeffreytang on 7/4/15.
 */

public class TranslateFromCaffe {

    public static void main(String args[]) throws IOException {
        final ObjectMapper MAPPER = new ObjectMapper(new ProtobufFactory());
        final ProtobufSchema caffeSchema = io.Reader.getCaffeSchema();


        ClassPathResource googleNetPath = new ClassPathResource("models/small_nin_imagenet/nin_imagenet_conv.caffemodel");
        byte[] googleNetSolverBytes = Files.readAllBytes(Paths.get(googleNetPath.getURI()));

//        List<String> googleNetSolverString = Files.readAllLines(Paths.get(googleNetSolverPath.getURI()));
        NetParameter netClass = MAPPER.readerFor(NetParameter.class).with(caffeSchema).readValue(googleNetSolverBytes);


    }

}
