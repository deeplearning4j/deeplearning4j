package translate;

/**
 * Created by jeffreytang on 7/7/15.
 */

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.dataformat.protobuf.ProtobufFactory;
import com.fasterxml.jackson.dataformat.protobuf.schema.ProtobufSchema;
import com.fasterxml.jackson.dataformat.protobuf.schema.ProtobufSchemaLoader;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;


import java.io.File;
import java.nio.charset.Charset;


public class exampleProtoBuf {

    final protected static String PROTOC_NAMED_STRINGS =
            "message NamedStrings {\n"
                    + " required string name = 2;\n"
                    + " repeated string values = 7;\n"
                    + "}\n";


    @AllArgsConstructor
    @NoArgsConstructor
    @Data
    static class NamedStrings {
        public String name;
    }

    static ObjectMapper MAPPER = new ObjectMapper(new ProtobufFactory());

//    public static void testStringArrayWithName() throws Exception {
//        ProtobufSchema schema = ProtobufSchemaLoader.std.parse(PROTOC_NAMED_STRINGS);
//        final ObjectWriter w = MAPPER.writerFor(NamedStrings.class).with(schema);
//        NamedStrings input = new NamedStrings("abc123");
//        w.writeValue(new File("out.json"), input);
//                byte[] readIn = java.nio.file.Files.readAllBytes(java.nio.file.Paths.get("out.json"));

//        return MAPPER.readerFor(NamedStrings.class).with(schema).readValue(readIn);
//    }

}
