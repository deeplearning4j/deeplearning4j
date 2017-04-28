package org.deeplearning4j.nearestneighbor.server;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.ObjectMapper;
import com.mashape.unirest.http.Unirest;
import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;

import org.deeplearning4j.nearestneighbor.model.NearestNeighborRequest;
import org.deeplearning4j.nearestneighbor.model.NearestNeighborsResult;
import org.deeplearning4j.nearestneighbor.model.NearstNeighborsResults;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.UUID;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 1/22/17.
 */
public class NearestNeighborsServerTest {

    private static NearestNeighborsServer server;
    private static File fileSave = new File(UUID.randomUUID().toString() + ".json");
    private static INDArray arr;
    @BeforeClass
    public static void before() throws Exception {
        server = new NearestNeighborsServer();
        arr = Nd4j.create(new double[][]{
                {1,2,3,4},
                {1,2,3,5},
                {3,4,5,6}
        });

        FileUtils.write(fileSave, Nd4jBase64.base64String(arr));
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
        server.runMain(new String[] {"--ndarrayPath", fileSave.getAbsolutePath(), "--nearestNeighborsPort", "9050"});
    }

    @AfterClass
    public static void after() throws Exception {
        fileSave.delete();
        server.stop();

    }

    @Test
    public void testServer() throws Exception {
        NearestNeighborRequest request = new NearestNeighborRequest();
        request.setInputIndex(0);
        request.setK(1);
        NearstNeighborsResults csvRecord = Unirest.post("http://localhost:9050/knn").header("accept", "application/json")
                .header("Content-Type", "application/json")
                .body(request)
                .asObject(NearstNeighborsResults.class).getBody();

        assertEquals(1,csvRecord.getResults().size());

    }

}
