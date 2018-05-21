package org.datavec.spark.transform;

import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.ObjectMapper;
import com.mashape.unirest.http.Unirest;
import org.apache.commons.io.FileUtils;
import org.datavec.image.transform.ImageTransformProcess;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchImageRecord;
import org.datavec.spark.transform.model.SingleImageRecord;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.serde.base64.Nd4jBase64;

import java.io.File;
import java.io.IOException;
import java.util.UUID;

import static org.junit.Assert.assertEquals;

/**
 * Created by kepricon on 17. 6. 19.
 */
public class ImageSparkTransformServerTest {

    private static ImageSparkTransformServer server;
    private static File fileSave = new File(UUID.randomUUID().toString() + ".json");

    @BeforeClass
    public static void before() throws Exception {
        server = new ImageSparkTransformServer();

        ImageTransformProcess imgTransformProcess = new ImageTransformProcess.Builder().seed(12345)
                        .scaleImageTransform(10).cropImageTransform(5).build();

        FileUtils.write(fileSave, imgTransformProcess.toJson());

        Unirest.setObjectMapper(new ObjectMapper() {
            private org.nd4j.shade.jackson.databind.ObjectMapper jacksonObjectMapper =
                            new org.nd4j.shade.jackson.databind.ObjectMapper();

            public <T> T readValue(String value, Class<T> valueType) {
                try {
                    return jacksonObjectMapper.readValue(value, valueType);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }

            public String writeValue(Object value) {
                try {
                    return jacksonObjectMapper.writeValueAsString(value);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        });

        server.runMain(new String[] {"--jsonPath", fileSave.getAbsolutePath(), "-dp", "9060"});
    }

    @AfterClass
    public static void after() throws Exception {
        fileSave.deleteOnExit();
        server.stop();

    }

    @Test
    public void testImageServer() throws Exception {
        SingleImageRecord record =
                        new SingleImageRecord(new ClassPathResource("testimages/class0/0.jpg").getFile().toURI());
        JsonNode jsonNode = Unirest.post("http://localhost:9060/transformincrementalarray")
                        .header("accept", "application/json").header("Content-Type", "application/json").body(record)
                        .asJson().getBody();
        Base64NDArrayBody array = Unirest.post("http://localhost:9060/transformincrementalarray")
                        .header("accept", "application/json").header("Content-Type", "application/json").body(record)
                        .asObject(Base64NDArrayBody.class).getBody();

        BatchImageRecord batch = new BatchImageRecord();
        batch.add(new ClassPathResource("testimages/class0/0.jpg").getFile().toURI());
        batch.add(new ClassPathResource("testimages/class0/1.png").getFile().toURI());
        batch.add(new ClassPathResource("testimages/class0/2.jpg").getFile().toURI());

        JsonNode jsonNodeBatch =
                        Unirest.post("http://localhost:9060/transformarray").header("accept", "application/json")
                                        .header("Content-Type", "application/json").body(batch).asJson().getBody();
        Base64NDArrayBody batchArray = Unirest.post("http://localhost:9060/transformarray")
                        .header("accept", "application/json").header("Content-Type", "application/json").body(batch)
                        .asObject(Base64NDArrayBody.class).getBody();

        INDArray result = getNDArray(jsonNode);
        assertEquals(1, result.size(0));

        INDArray batchResult = getNDArray(jsonNodeBatch);
        assertEquals(3, batchResult.size(0));

        System.out.println(array);
    }

    @Test
    public void testImageServerMultipart() throws Exception {
        JsonNode jsonNode = Unirest.post("http://localhost:9060/transformimage")
                .header("accept", "application/json")
                .field("file1", new ClassPathResource("testimages/class0/0.jpg").getFile())
                .field("file2", new ClassPathResource("testimages/class0/1.png").getFile())
                .field("file3", new ClassPathResource("testimages/class0/2.jpg").getFile())
                .asJson().getBody();


        INDArray batchResult = getNDArray(jsonNode);
        assertEquals(3, batchResult.size(0));

        System.out.println(batchResult);
    }

    @Test
    public void testImageServerSingleMultipart() throws Exception {
        JsonNode jsonNode = Unirest.post("http://localhost:9060/transformimage")
                .header("accept", "application/json")
                .field("file1", new ClassPathResource("testimages/class0/0.jpg").getFile())
                .asJson().getBody();


        INDArray result = getNDArray(jsonNode);
        assertEquals(1, result.size(0));

        System.out.println(result);
    }

    public INDArray getNDArray(JsonNode node) throws IOException {
        return Nd4jBase64.fromBase64(node.getObject().getString("ndarray"));
    }
}
