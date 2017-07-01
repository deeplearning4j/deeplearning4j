package org.deeplearning4j.nearestneighbor.server;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.mashape.unirest.http.ObjectMapper;
import com.mashape.unirest.http.Unirest;
import org.apache.commons.io.FileUtils;

import org.apache.commons.io.IOUtils;
import org.deeplearning4j.nearestneighbor.client.NearestNeighborsClient;
import org.deeplearning4j.nearestneighbor.model.Base64NDArrayBody;
import org.deeplearning4j.nearestneighbor.model.NearestNeighborRequest;
import org.deeplearning4j.nearestneighbor.model.NearstNeighborsResults;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;
import org.nd4j.serde.binary.BinarySerde;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
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

        FileOutputStream fileOutputStream = new FileOutputStream(fileSave);
        ByteBuffer byteBuffer = BinarySerde.toByteBuffer(arr);
        byte[] allBuffer = new byte[byteBuffer.capacity()];
        byteBuffer.get(allBuffer);
        IOUtils.write(allBuffer,fileOutputStream);
        fileOutputStream.flush();
        fileOutputStream.close();
        server.runMain(new String[] {"--ndarrayPath", fileSave.getAbsolutePath(), "--nearestNeighborsPort", "9050"});
    }

    @AfterClass
    public static void after() throws Exception {
        fileSave.delete();
        server.stop();

    }

    @Test
    public void testServer() throws Exception {
        NearestNeighborsClient nearestNeighborsClient = new NearestNeighborsClient("http://localhost:9050");
        NearstNeighborsResults csvRecord = nearestNeighborsClient.knn(0,1);

        assertEquals(1,csvRecord.getResults().size());

        NearstNeighborsResults csvRecord2 = nearestNeighborsClient.knnNew(1,arr.slice(0));
        assertEquals(1,csvRecord2.getResults().size());


    }

}
