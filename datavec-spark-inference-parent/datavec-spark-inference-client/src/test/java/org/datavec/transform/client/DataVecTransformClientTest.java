package org.datavec.transform.client;

import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.spark.transform.CSVSparkTransformServer;
import org.datavec.spark.transform.client.DataVecTransformClient;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchCSVRecord;
import org.datavec.spark.transform.model.SingleCSVRecord;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.serde.base64.Nd4jBase64;

import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;
import java.util.Arrays;
import java.util.UUID;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeNotNull;

/**
 * Created by agibsonccc on 6/12/17.
 */
public class DataVecTransformClientTest {
    private static CSVSparkTransformServer server;
    private static int port = getAvailablePort();
    private static DataVecTransformClient client;
    private static Schema schema = new Schema.Builder().addColumnDouble("1.0")
            .addColumnDouble("2.0").build();
    private static TransformProcess transformProcess =
            new TransformProcess.Builder(schema)
                    .convertToString("1.0")
                    .convertToString("2.0").build();
    private static File fileSave = new File(UUID.randomUUID().toString() + ".json");

    @BeforeClass
    public static void beforeClass() throws Exception {
        FileUtils.write(fileSave, transformProcess.toJson());
        fileSave.deleteOnExit();
        server = new CSVSparkTransformServer();
        server.runMain(new String[]{
                "-dp", String.valueOf(port)
        });

        client = new DataVecTransformClient("http://localhost:" + port);
        client.setCSVTransformProcess(transformProcess);
    }

    @AfterClass
    public static void afterClass() throws Exception {
        server.stop();
    }

    @Test
    public void testRecord() throws Exception {
        SingleCSVRecord singleCsvRecord = new SingleCSVRecord(new String[]{"0","0"});
        SingleCSVRecord transformed = client.transformIncremental(singleCsvRecord);
        assertEquals(singleCsvRecord.getValues().length,transformed.getValues().length);
        Base64NDArrayBody body = client.transformArrayIncremental(singleCsvRecord);
        INDArray arr = Nd4jBase64.fromBase64(body.getNdarray());
        assumeNotNull(arr);
    }

    @Test
    public void testBatchRecord() throws Exception {
        SingleCSVRecord singleCsvRecord = new SingleCSVRecord(new String[]{"0","0"});

        BatchCSVRecord batchCSVRecord = new BatchCSVRecord(Arrays.asList(singleCsvRecord, singleCsvRecord));
        BatchCSVRecord batchCSVRecord1 = client.transform(batchCSVRecord);
        assertEquals(batchCSVRecord.getRecords().size(), batchCSVRecord1.getRecords().size());

        Base64NDArrayBody body = client.transformArray(batchCSVRecord);
        INDArray arr = Nd4jBase64.fromBase64(body.getNdarray());
        assumeNotNull(arr);
    }





    public static int getAvailablePort() {
        try {
            ServerSocket socket = new ServerSocket(0);
            try {
                return socket.getLocalPort();
            } finally {
                socket.close();
            }
        } catch (IOException e) {
            throw new IllegalStateException("Cannot find available port: " + e.getMessage(), e);
        }
    }

}
