package org.datavec.spark.transform;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.ObjectMapper;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.options.Option;
import com.mashape.unirest.http.options.Options;
import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchRecord;
import org.datavec.spark.transform.model.CSVRecord;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import play.libs.Json;

import java.io.File;
import java.io.IOException;
import java.util.UUID;

/**
 * Created by agibsonccc on 1/22/17.
 */
public class CSVSparkTransformServerTest {

    private static CSVSparkTransformServer server;
    private static Schema schema = new Schema.Builder().addColumnDouble("1.0").addColumnDouble("2.0").build();
    private static TransformProcess transformProcess =
                    new TransformProcess.Builder(schema).convertToString("1.0").convertToString("2.0").build();
    private static File fileSave = new File(UUID.randomUUID().toString() + ".json");

    @BeforeClass
    public static void before() throws Exception {
        server = new CSVSparkTransformServer();
        FileUtils.write(fileSave, transformProcess.toJson());
        // Only one time
        Unirest.setObjectMapper(new ObjectMapper() {
            private com.fasterxml.jackson.databind.ObjectMapper jacksonObjectMapper =
                            new com.fasterxml.jackson.databind.ObjectMapper();

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
                } catch (JsonProcessingException e) {
                    throw new RuntimeException(e);
                }
            }
        });
        server.runMain(new String[] {"--jsonPath", fileSave.getAbsolutePath(), "-dp", "9050"});
    }

    @AfterClass
    public static void after() throws Exception {
        fileSave.delete();
        server.stop();

    }




    @Test
    public void testServer() throws Exception {
        String[] values = new String[] {"1.0", "2.0"};
        CSVRecord record = new CSVRecord(values);
        JsonNode jsonNode = Unirest.post("http://localhost:9050/transformincremental").header("accept", "application/json")
                        .header("Content-Type", "application/json").body(record).asJson().getBody();
        CSVRecord csvRecord = Unirest.post("http://localhost:9050/transformincremental").header("accept", "application/json")
                        .header("Content-Type", "application/json").body(record).asObject(CSVRecord.class).getBody();

        BatchRecord batchRecord = new BatchRecord();
        for (int i = 0; i < 3; i++)
            batchRecord.add(csvRecord);
        BatchRecord batchRecord1 = Unirest.post("http://localhost:9050/transform")
                        .header("accept", "application/json").header("Content-Type", "application/json")
                        .body(batchRecord).asObject(BatchRecord.class).getBody();

        Base64NDArrayBody array = Unirest.post("http://localhost:9050/transformedincrementalarray")
                        .header("accept", "application/json").header("Content-Type", "application/json").body(record)
                        .asObject(Base64NDArrayBody.class).getBody();

        Base64NDArrayBody batchArray1 = Unirest.post("http://localhost:9050/transformedarray")
                        .header("accept", "application/json").header("Content-Type", "application/json")
                        .body(batchRecord).asObject(Base64NDArrayBody.class).getBody();



    }

}
