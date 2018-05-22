package org.datavec.spark.transform;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.ObjectMapper;
import com.mashape.unirest.http.Unirest;
import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchCSVRecord;
import org.datavec.spark.transform.model.SingleCSVRecord;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.UUID;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeNotNull;

/**
 * Created by agibsonccc on 1/22/17.
 */
public class CSVSparkTransformServerNoJsonTest {

    private static CSVSparkTransformServer server;
    private static Schema schema = new Schema.Builder().addColumnDouble("1.0").addColumnDouble("2.0").build();
    private static TransformProcess transformProcess =
                    new TransformProcess.Builder(schema).convertToDouble("1.0").convertToDouble("2.0").build();
    private static File fileSave = new File(UUID.randomUUID().toString() + ".json");

    @BeforeClass
    public static void before() throws Exception {
        server = new CSVSparkTransformServer();
        FileUtils.write(fileSave, transformProcess.toJson());
        // Only one time
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
        server.runMain(new String[] {"-dp", "9050"});
    }

    @AfterClass
    public static void after() throws Exception {
        fileSave.delete();
        server.stop();

    }



    @Test
    public void testServer() throws Exception {
        assertTrue(server.getTransform() == null);
        JsonNode jsonStatus = Unirest.post("http://localhost:9050/transformprocess")
                        .header("accept", "application/json").header("Content-Type", "application/json")
                        .body(transformProcess.toJson()).asJson().getBody();
        assumeNotNull(server.getTransform());

        String[] values = new String[] {"1.0", "2.0"};
        SingleCSVRecord record = new SingleCSVRecord(values);
        JsonNode jsonNode =
                        Unirest.post("http://localhost:9050/transformincremental").header("accept", "application/json")
                                        .header("Content-Type", "application/json").body(record).asJson().getBody();
        SingleCSVRecord singleCsvRecord = Unirest.post("http://localhost:9050/transformincremental")
                        .header("accept", "application/json").header("Content-Type", "application/json").body(record)
                        .asObject(SingleCSVRecord.class).getBody();

        BatchCSVRecord batchCSVRecord = new BatchCSVRecord();
        for (int i = 0; i < 3; i++)
            batchCSVRecord.add(singleCsvRecord);
    /*    BatchCSVRecord batchCSVRecord1 = Unirest.post("http://localhost:9050/transform")
                        .header("accept", "application/json").header("Content-Type", "application/json")
                        .body(batchCSVRecord).asObject(BatchCSVRecord.class).getBody();

        Base64NDArrayBody array = Unirest.post("http://localhost:9050/transformincrementalarray")
                        .header("accept", "application/json").header("Content-Type", "application/json").body(record)
                        .asObject(Base64NDArrayBody.class).getBody();
*/
        Base64NDArrayBody batchArray1 = Unirest.post("http://localhost:9050/transformarray")
                        .header("accept", "application/json").header("Content-Type", "application/json")
                        .body(batchCSVRecord).asObject(Base64NDArrayBody.class).getBody();



    }

}
