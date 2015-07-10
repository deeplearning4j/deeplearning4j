package translate;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.dataformat.protobuf.ProtobufFactory;
import com.fasterxml.jackson.dataformat.protobuf.schema.ProtobufSchema;
import com.fasterxml.jackson.dataformat.protobuf.schema.ProtobufSchemaLoader;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Created by jeffreytang on 7/6/15.
 */
public class JacksonProtoBuf {
    // Schema Class
    static class NamedStrings {
        public String name;

        public NamedStrings() {
        }

        public NamedStrings(String n) {
            name = n;
        }
    }

    // Proto Schema
    final protected static String PROTOC_NAMED_STRINGS =
            "message NamedStrings {\n"
                    + " required string name = 2;\n"
                    + "}\n";
    // Proto Mapper
    final static ObjectMapper MAPPER = new ObjectMapper(new ProtobufFactory());


    // Main method
    public static void main(String args[]) throws IOException {

        ProtobufSchema schema = ProtobufSchemaLoader.std.parse(PROTOC_NAMED_STRINGS);

        NamedStrings input = new NamedStrings("testing");
        final ObjectWriter w = MAPPER.writerFor(NamedStrings.class).with(schema);
        byte[] bytes = w.writeValueAsBytes(input);
//        byte[] byteArr = Files.readAllBytes(Paths.get(new ClassPathResource("test/test_case.json").getURI()));

        NamedStrings testClass = MAPPER.readerFor(NamedStrings.class).with(schema).readValue(bytes);


    }
}
