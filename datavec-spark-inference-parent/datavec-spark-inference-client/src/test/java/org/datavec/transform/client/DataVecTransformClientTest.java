package org.datavec.transform.client;

import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.spark.transform.CSVSparkTransformServer;
import org.datavec.spark.transform.client.DataVecTransformClient;
import org.datavec.spark.transform.model.CSVRecord;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;
import java.util.UUID;

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
        client.setTransformProcess(transformProcess);
    }

    @AfterClass
    public static void afterClass() throws Exception {
        server.stop();
    }

    @Test
    public void testBatches() {
        DataSet dataSet = new DataSet(Nd4j.create(2, 2), Nd4j.create(new double[][]
                {{1, 1}, {1, 1}}));
        CSVRecord csvRecord = CSVRecord.fromRow(dataSet.get(0));
        client.transformIncremental(csvRecord);
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
