/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.spark.transform;

import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.ObjectMapper;
import com.mashape.unirest.http.Unirest;
import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.image.transform.ImageTransformProcess;
import org.datavec.spark.transform.model.*;
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
 * Created by kepricon on 17. 6. 20.
 */
public class SparkTransformServerTest {
    private static SparkTransformServerChooser serverChooser;
    private static Schema schema = new Schema.Builder().addColumnDouble("1.0").addColumnDouble("2.0").build();
    private static TransformProcess transformProcess =
                    new TransformProcess.Builder(schema).convertToDouble("1.0").convertToDouble(    "2.0").build();

    private static File imageTransformFile = new File(UUID.randomUUID().toString() + ".json");
    private static File csvTransformFile = new File(UUID.randomUUID().toString() + ".json");

    @BeforeClass
    public static void before() throws Exception {
        serverChooser = new SparkTransformServerChooser();

        ImageTransformProcess imgTransformProcess = new ImageTransformProcess.Builder().seed(12345)
                        .scaleImageTransform(10).cropImageTransform(5).build();

        FileUtils.write(imageTransformFile, imgTransformProcess.toJson());

        FileUtils.write(csvTransformFile, transformProcess.toJson());

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


    }

    @AfterClass
    public static void after() throws Exception {
        imageTransformFile.deleteOnExit();
        csvTransformFile.deleteOnExit();
    }

    @Test
    public void testImageServer() throws Exception {
        serverChooser.runMain(new String[] {"--jsonPath", imageTransformFile.getAbsolutePath(), "-dp", "9060", "-dt",
                        TransformDataType.IMAGE.toString()});

        SingleImageRecord record =
                        new SingleImageRecord(new ClassPathResource("datavec-spark-inference/testimages/class0/0.jpg").getFile().toURI());
        JsonNode jsonNode = Unirest.post("http://localhost:9060/transformincrementalarray")
                        .header("accept", "application/json").header("Content-Type", "application/json").body(record)
                        .asJson().getBody();
        Base64NDArrayBody array = Unirest.post("http://localhost:9060/transformincrementalarray")
                        .header("accept", "application/json").header("Content-Type", "application/json").body(record)
                        .asObject(Base64NDArrayBody.class).getBody();

        BatchImageRecord batch = new BatchImageRecord();
        batch.add(new ClassPathResource("datavec-spark-inference/testimages/class0/0.jpg").getFile().toURI());
        batch.add(new ClassPathResource("datavec-spark-inference/testimages/class0/1.png").getFile().toURI());
        batch.add(new ClassPathResource("datavec-spark-inference/testimages/class0/2.jpg").getFile().toURI());

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

        serverChooser.getSparkTransformServer().stop();
    }

    @Test
    public void testCSVServer() throws Exception {
        serverChooser.runMain(new String[] {"--jsonPath", csvTransformFile.getAbsolutePath(), "-dp", "9050", "-dt",
                        TransformDataType.CSV.toString()});

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
        BatchCSVRecord batchCSVRecord1 = Unirest.post("http://localhost:9050/transform")
                        .header("accept", "application/json").header("Content-Type", "application/json")
                        .body(batchCSVRecord).asObject(BatchCSVRecord.class).getBody();

        Base64NDArrayBody array = Unirest.post("http://localhost:9050/transformincrementalarray")
                        .header("accept", "application/json").header("Content-Type", "application/json").body(record)
                        .asObject(Base64NDArrayBody.class).getBody();

        Base64NDArrayBody batchArray1 = Unirest.post("http://localhost:9050/transformarray")
                        .header("accept", "application/json").header("Content-Type", "application/json")
                        .body(batchCSVRecord).asObject(Base64NDArrayBody.class).getBody();


        serverChooser.getSparkTransformServer().stop();
    }

    public INDArray getNDArray(JsonNode node) throws IOException {
        return Nd4jBase64.fromBase64(node.getObject().getString("ndarray"));
    }
}
