package org.datavec.spark.transform;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.ObjectMapper;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.apache.commons.io.FileUtils;
import org.datavec.image.transform.ImageTransformProcess;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.CSVRecord;
import org.datavec.spark.transform.model.ImageRecord;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.UUID;

/**
 * Created by kepricon on 17. 6. 19.
 */
public class ImageSparkTransformServerTest {

    private static ImageSparkTransformServer server;
    private static File fileSave = new File(UUID.randomUUID().toString() + ".json");

    @BeforeClass
    public static void before() throws Exception {
        server = new ImageSparkTransformServer();

        ImageTransformProcess imgTransformProcess = new ImageTransformProcess.Builder()
                .seed(12345)
                .scaleImageTransform(10)
                .cropImageTransform(5)
                .build();

        FileUtils.write(fileSave, imgTransformProcess.toJson());

        Unirest.setObjectMapper(new ObjectMapper() {
            private com.fasterxml.jackson.databind.ObjectMapper jacksonObjectMapper =
                    new com.fasterxml.jackson.databind.ObjectMapper();

            public <T> T readValue(String value, Class<T> valueType) {
                try {
                    System.out.println("value : " + value);
                    System.out.println("valueType : " + valueType);
                    return jacksonObjectMapper.readValue(value, valueType);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }

            public String writeValue(Object value) {
                try {
                    return jacksonObjectMapper.writeValueAsString(value);
                } catch (JsonProcessingException e) {
                    throw new RuntimeException(e);
                }
            }
        });

        server.runMain(new String[] {"--jsonPath", fileSave.getAbsolutePath(), "-dp", "9060"});
    }

    @AfterClass
    public static void after() throws Exception {
        fileSave.delete();
        server.stop();

    }

    @Test
    public void testImageServer() throws Exception {
        ImageRecord record = new ImageRecord(new File("/home/kepricon/git/DataVec/datavec-spark-inference-parent/datavec-spark-inference-model/src/test/resources/testimages/class0/0.jpg").toURI());
        JsonNode jsonNode = Unirest.post("http://localhost:9060/transformincrementalarray").header("accept", "application/json")
                .header("Content-Type", "application/json").body(record).asJson().getBody();
        Base64NDArrayBody array = Unirest.post("http://localhost:9060/transformincrementalarray").header("accept", "application/json")
                .header("Content-Type", "application/json").body(record).asObject(Base64NDArrayBody.class).getBody();

        System.out.println(jsonNode);
        System.out.println(array);
    }
}
