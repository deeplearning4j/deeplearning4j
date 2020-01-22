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

package org.deeplearning4j.remote;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.parallelism.inference.InferenceMode;
import org.deeplearning4j.remote.helpers.House;
import org.deeplearning4j.remote.helpers.HouseToPredictedPriceAdapter;
import org.deeplearning4j.remote.helpers.PredictedPrice;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.adapters.InferenceAdapter;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.remote.clients.JsonRemoteInference;
import org.nd4j.remote.clients.serde.JsonDeserializer;
import org.nd4j.remote.clients.serde.JsonSerializer;
import org.nd4j.shade.jackson.databind.ObjectMapper;


import java.io.IOException;
import java.util.Collections;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.deeplearning4j.parallelism.inference.InferenceMode.INPLACE;
import static org.deeplearning4j.parallelism.inference.InferenceMode.SEQUENTIAL;
import static org.junit.Assert.*;

@Slf4j
public class JsonModelServerTest extends BaseDL4JTest {
    private static final MultiLayerNetwork model;

    static {
        val conf = new NeuralNetConfiguration.Builder()
                .seed(119)
                .updater(new Adam(0.119f))
                .weightInit(WeightInit.XAVIER)
                .list()
                    .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(4).nOut(10).build())
                    .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).activation(Activation.SIGMOID).nIn(10).nOut(1).build())
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
    }

    @After
    public void pause() throws Exception {
        // Need to wait for server shutdown; without sleep, tests will fail if starting immediately after shutdown
        TimeUnit.SECONDS.sleep(2);
    }

    private AtomicInteger portCount = new AtomicInteger(18080);
    private int PORT;

    @Before
    public void setPort(){
        PORT = portCount.getAndIncrement();
    }


    @Test
    public void testStartStopParallel() throws Exception {
        val sd = SameDiff.create();
        val sdVariable = sd.placeHolder("input", DataType.INT, 1,4);
        val result = sdVariable.add(1.0);
        val total = result.mean("total", Integer.MAX_VALUE);

        val serverDL = new JsonModelServer.Builder<House, PredictedPrice>(model)
                .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                .inputDeserializer(new House.HouseDeserializer())
                .numWorkers(1)
                .inferenceMode(SEQUENTIAL)
                .inferenceAdapter(new HouseToPredictedPriceAdapter())
                .port(PORT)
                .build();

        val serverSD = new JsonModelServer.Builder<House, PredictedPrice>(sd)
                .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                .inputDeserializer(new House.HouseDeserializer())
                .inferenceAdapter(new HouseToPredictedPriceAdapter())
                .orderedInputNodes(new String[]{"input"})
                .orderedOutputNodes(new String[]{"total"})
                .port(PORT+1)
                .build();
        try {
            serverDL.start();
            serverSD.start();

            val clientDL = JsonRemoteInference.<House, PredictedPrice>builder()
                    .inputSerializer(new House.HouseSerializer())
                    .outputDeserializer(new PredictedPrice.PredictedPriceDeserializer())
                    .endpointAddress("http://localhost:" + PORT  + "/v1/serving")
                    .build();

            int district = 2;
            House house = House.builder().area(100).bathrooms(2).bedrooms(3).district(district).build();
            PredictedPrice price = clientDL.predict(house);
            long timeStart = System.currentTimeMillis();
            price = clientDL.predict(house);
            long timeStop = System.currentTimeMillis();
            log.info("Time spent: {} ms", timeStop - timeStart);
            assertNotNull(price);
            assertEquals((float) 0.421444, price.getPrice(), 1e-5);

            val clientSD = JsonRemoteInference.<House, PredictedPrice>builder()
                    .inputSerializer(new House.HouseSerializer())
                    .outputDeserializer(new PredictedPrice.PredictedPriceDeserializer())
                    .endpointAddress("http://localhost:" + (PORT+1)  + "/v1/serving")
                    .build();

            PredictedPrice price2 = clientSD.predict(house);
            timeStart = System.currentTimeMillis();
            price = clientSD.predict(house);
            timeStop = System.currentTimeMillis();
            log.info("Time spent: {} ms", timeStop - timeStart);
            assertNotNull(price);
            assertEquals((float) 3.0, price.getPrice(), 1e-5);

        }
        finally {
            serverSD.stop();
            serverDL.stop();
        }
    }

    @Test
    public void testStartStopSequential() throws Exception {
        val sd = SameDiff.create();
        val sdVariable = sd.placeHolder("input", DataType.INT, 1,4);
        val result = sdVariable.add(1.0);
        val total = result.mean("total", Integer.MAX_VALUE);

        val serverDL = new JsonModelServer.Builder<House, PredictedPrice>(model)
                .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                .inputDeserializer(new House.HouseDeserializer())
                .numWorkers(1)
                .inferenceMode(SEQUENTIAL)
                .inferenceAdapter(new HouseToPredictedPriceAdapter())
                .port(PORT)
                .build();

        val serverSD = new JsonModelServer.Builder<House, PredictedPrice>(sd)
                .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                .inputDeserializer(new House.HouseDeserializer())
                .inferenceAdapter(new HouseToPredictedPriceAdapter())
                .orderedInputNodes(new String[]{"input"})
                .orderedOutputNodes(new String[]{"total"})
                .port(PORT+1)
                .build();

        serverDL.start();
        serverDL.stop();

        serverSD.start();
        serverSD.stop();
    }

    @Test
    public void basicServingTestForSD() throws Exception {
        val sd = SameDiff.create();
        val sdVariable = sd.placeHolder("input", DataType.INT, 1,4);
        val result = sdVariable.add(1.0);
        val total = result.mean("total", Integer.MAX_VALUE);

        val server = new JsonModelServer.Builder<House, PredictedPrice>(sd)
                .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                .inputDeserializer(new House.HouseDeserializer())
                .inferenceAdapter(new HouseToPredictedPriceAdapter())
                .orderedInputNodes(new String[]{"input"})
                .orderedOutputNodes(new String[]{"total"})
                .port(PORT)
                .build();

        try {
            server.start();

            val client = JsonRemoteInference.<House, PredictedPrice>builder()
                    .inputSerializer(new House.HouseSerializer())
                    .outputDeserializer(new PredictedPrice.PredictedPriceDeserializer())
                    .endpointAddress("http://localhost:" + PORT + "/v1/serving")
                    .build();

            int district = 2;
            House house = House.builder().area(100).bathrooms(2).bedrooms(3).district(district).build();

            // warmup
            PredictedPrice price = client.predict(house);

            val timeStart = System.currentTimeMillis();
            price = client.predict(house);
            val timeStop = System.currentTimeMillis();

            log.info("Time spent: {} ms", timeStop - timeStart);

            assertNotNull(price);
            assertEquals((float) district + 1.0f, price.getPrice(), 1e-5);
        }
        finally {
            server.stop();
        }
    }

    @Test
    public void basicServingTestForDLSynchronized() throws Exception {
        val server = new JsonModelServer.Builder<House, PredictedPrice>(model)
                .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                .inputDeserializer(new House.HouseDeserializer())
                .numWorkers(1)
                .inferenceMode(INPLACE)
                .inferenceAdapter(new HouseToPredictedPriceAdapter())
                .port(PORT)
                .build();

        try {
            server.start();

            val client = JsonRemoteInference.<House, PredictedPrice>builder()
                    .inputSerializer(new House.HouseSerializer())
                    .outputDeserializer(new PredictedPrice.PredictedPriceDeserializer())
                    .endpointAddress("http://localhost:" + PORT  + "/v1/serving")
                    .build();

            int district = 2;
            House house1 = House.builder().area(100).bathrooms(2).bedrooms(3).district(district).build();
            House house2 = House.builder().area(50).bathrooms(1).bedrooms(2).district(district).build();
            House house3 = House.builder().area(80).bathrooms(1).bedrooms(3).district(district).build();

            // warmup
            PredictedPrice price = client.predict(house1);

            val timeStart = System.currentTimeMillis();
            PredictedPrice price1 = client.predict(house1);
            PredictedPrice price2 = client.predict(house2);
            PredictedPrice price3 = client.predict(house3);
            val timeStop = System.currentTimeMillis();

            log.info("Time spent: {} ms", timeStop - timeStart);

            assertNotNull(price);
            assertEquals((float) 0.421444, price.getPrice(), 1e-5);

        } finally {
            server.stop();
        }
    }

    @Test
    public void basicServingTestForDL() throws Exception {

        val server = new JsonModelServer.Builder<House, PredictedPrice>(model)
                .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                .inputDeserializer(new House.HouseDeserializer())
                .numWorkers(1)
                .inferenceMode(SEQUENTIAL)
                .inferenceAdapter(new HouseToPredictedPriceAdapter())
                .port(PORT)
                .parallelMode(false)
                .build();

        try {
            server.start();

            val client = JsonRemoteInference.<House, PredictedPrice>builder()
                    .inputSerializer(new House.HouseSerializer())
                    .outputDeserializer(new PredictedPrice.PredictedPriceDeserializer())
                    .endpointAddress("http://localhost:" + PORT  + "/v1/serving")
                    .build();

            int district = 2;
            House house = House.builder().area(100).bathrooms(2).bedrooms(3).district(district).build();

            // warmup
            PredictedPrice price = client.predict(house);

            val timeStart = System.currentTimeMillis();
            price = client.predict(house);
            val timeStop = System.currentTimeMillis();

            log.info("Time spent: {} ms", timeStop - timeStart);

            assertNotNull(price);
            assertEquals((float) 0.421444, price.getPrice(), 1e-5);

        } finally {
            server.stop();
        }
    }

    @Test
    public void testDeserialization_1() {
        String request = "{\"bedrooms\":3,\"area\":100,\"district\":2,\"bathrooms\":2}";
        val deserializer = new House.HouseDeserializer();
        val result = deserializer.deserialize(request);
        assertEquals(2, result.getDistrict());
        assertEquals(100, result.getArea());
        assertEquals(2, result.getBathrooms());
        assertEquals(3, result.getBedrooms());

    }

    @Test
    public void testDeserialization_2() {
        String request = "{\"price\":1}";
        val deserializer = new PredictedPrice.PredictedPriceDeserializer();
        val result = deserializer.deserialize(request);
        assertEquals(1.0, result.getPrice(), 1e-4);
    }

    @Test(expected = NullPointerException.class)
    public void negativeServingTest_1() throws Exception {

        val server = new JsonModelServer.Builder<House, PredictedPrice>(model)
                .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                .inputDeserializer(null)
                .port(PORT)
                .build();
    }

    @Test //(expected = NullPointerException.class)
    public void negativeServingTest_2() throws Exception {

        val server = new JsonModelServer.Builder<House, PredictedPrice>(model)
                .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                .inputDeserializer(new House.HouseDeserializer())
                .inferenceAdapter(new HouseToPredictedPriceAdapter())
                .port(PORT)
                .build();

    }

    @Test(expected = IOException.class)
    public void negativeServingTest_3() throws Exception {

        val server = new JsonModelServer.Builder<House, PredictedPrice>(model)
                    .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                    .inputDeserializer(new House.HouseDeserializer())
                    .inferenceAdapter(new HouseToPredictedPriceAdapter())
                    .inferenceMode(SEQUENTIAL)
                    .numWorkers(1)
                    .port(PORT)
                    .build();

        try {
            server.start();

            val client = JsonRemoteInference.<House, PredictedPrice>builder()
                    .inputSerializer(new House.HouseSerializer())
                    .outputDeserializer(new JsonDeserializer<PredictedPrice>() {
                        @Override
                        public PredictedPrice deserialize(String json) {
                            return null;
                        }
                    })
                    .endpointAddress("http://localhost:" + PORT + "/v1/serving")
                    .build();

            int district = 2;
            House house = House.builder().area(100).bathrooms(2).bedrooms(3).district(district).build();

            // warmup
            PredictedPrice price = client.predict(house);
        } finally {
            server.stop();
        }
    }

    @Test
    public void asyncServingTest() throws Exception {

        val server = new JsonModelServer.Builder<House, PredictedPrice>(model)
                    .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                    .inputDeserializer(new House.HouseDeserializer())
                    .inferenceAdapter(new HouseToPredictedPriceAdapter())
                    .inferenceMode(SEQUENTIAL)
                    .numWorkers(1)
                    .port(PORT)
                    .build();

        try {
            server.start();

            val client = JsonRemoteInference.<House, PredictedPrice>builder()
                    .inputSerializer(new House.HouseSerializer())
                    .outputDeserializer(new PredictedPrice.PredictedPriceDeserializer())
                    .endpointAddress("http://localhost:" + PORT + "/v1/serving")
                    .build();

            int district = 2;
            House house = House.builder().area(100).bathrooms(2).bedrooms(3).district(district).build();

            val timeStart = System.currentTimeMillis();
            Future<PredictedPrice> price = client.predictAsync(house);
            assertNotNull(price);
            assertEquals((float) 0.421444, price.get().getPrice(), 1e-5);
            val timeStop = System.currentTimeMillis();

            log.info("Time spent: {} ms", timeStop - timeStart);
        }
        finally {
            server.stop();
        }
    }

    @Test
    public void negativeAsyncTest() throws Exception {

        val server = new JsonModelServer.Builder<House, PredictedPrice>(model)
                .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                .inputDeserializer(new House.HouseDeserializer())
                .inferenceAdapter(new HouseToPredictedPriceAdapter())
                .inferenceMode(InferenceMode.BATCHED)
                .numWorkers(1)
                .port(PORT)
                .build();

        try {
            server.start();

            // Fake deserializer to test failure
            val client = JsonRemoteInference.<House, PredictedPrice>builder()
                    .inputSerializer(new House.HouseSerializer())
                    .outputDeserializer(new JsonDeserializer<PredictedPrice>() {
                        @Override
                        public PredictedPrice deserialize(String json) {
                            return null;
                        }
                    })
                    .endpointAddress("http://localhost:" + PORT + "/v1/serving")
                    .build();

            int district = 2;
            House house = House.builder().area(100).bathrooms(2).bedrooms(3).district(district).build();

            val timeStart = System.currentTimeMillis();
            try {
                Future<PredictedPrice> price = client.predictAsync(house);
                assertNotNull(price);
                assertEquals((float) district + 1.0f, price.get().getPrice(), 1e-5);
                val timeStop = System.currentTimeMillis();

                log.info("Time spent: {} ms", timeStop - timeStart);
            } catch (ExecutionException e) {
                assertTrue(e.getMessage().contains("Deserialization failed"));
            }
        } finally {
            server.stop();
        }
    }


    @Test
    public void testSameDiffMnist() throws Exception {

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 28*28);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 28*28, 10));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 1, 10));
        SDVariable sm = sd.nn.softmax("softmax", in.mmul(w).add(b));

        val server = new JsonModelServer.Builder<float[], Integer>(sd)
                .outputSerializer( new IntSerde())
                .inputDeserializer(new FloatSerde())
                .inferenceAdapter(new InferenceAdapter<float[], Integer>() {
                    @Override
                    public MultiDataSet apply(float[] input) {
                        return new MultiDataSet(Nd4j.create(input, 1, input.length), null);
                    }

                    @Override
                    public Integer apply(INDArray... nnOutput) {
                        return nnOutput[0].argMax().getInt(0);
                    }
                })
                .orderedInputNodes("in")
                .orderedOutputNodes("softmax")
                .port(PORT+1)
                .build();

        val client = JsonRemoteInference.<float[], Integer>builder()
                .endpointAddress("http://localhost:" + (PORT+1) + "/v1/serving")
                .outputDeserializer(new IntSerde())
                .inputSerializer( new FloatSerde())
                .build();

        try{
            server.start();
            for( int i=0; i<10; i++ ){
                INDArray f = Nd4j.rand(DataType.FLOAT, 1, 28*28);
                INDArray exp = sd.output(Collections.singletonMap("in", f), "softmax").get("softmax");
                float[] fArr = f.toFloatVector();
                int out = client.predict(fArr);
                assertEquals(exp.argMax().getInt(0), out);
            }
        } finally {
            server.stop();
        }
    }

    @Test
    public void testMlnMnist() throws Exception {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new DenseLayer.Builder().nIn(784).nOut(10).build())
                .layer(new LossLayer.Builder().activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        val server = new JsonModelServer.Builder<float[], Integer>(net)
                .outputSerializer( new IntSerde())
                .inputDeserializer(new FloatSerde())
                .inferenceAdapter(new InferenceAdapter<float[], Integer>() {
                    @Override
                    public MultiDataSet apply(float[] input) {
                        return new MultiDataSet(Nd4j.create(input, 1, input.length), null);
                    }

                    @Override
                    public Integer apply(INDArray... nnOutput) {
                        return nnOutput[0].argMax().getInt(0);
                    }
                })
                .orderedInputNodes("in")
                .orderedOutputNodes("softmax")
                .port(PORT + 1)
                .inferenceMode(SEQUENTIAL)
                .numWorkers(2)
                .build();

        val client = JsonRemoteInference.<float[], Integer>builder()
                .endpointAddress("http://localhost:" + (PORT + 1) + "/v1/serving")
                .outputDeserializer(new IntSerde())
                .inputSerializer( new FloatSerde())
                .build();

        try {
            server.start();
            for (int i = 0; i < 10; i++) {
                INDArray f = Nd4j.rand(DataType.FLOAT, 1, 28 * 28);
                INDArray exp = net.output(f);
                float[] fArr = f.toFloatVector();
                int out = client.predict(fArr);
                assertEquals(exp.argMax().getInt(0), out);
            }
        } catch (Exception e){
            e.printStackTrace();
            throw e;
        } finally {
            server.stop();
        }
    }

    @Test
    public void testCompGraph() throws Exception {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input1", "input2")
                .addLayer("L1", new DenseLayer.Builder().nIn(3).nOut(4).build(), "input1")
                .addLayer("L2", new DenseLayer.Builder().nIn(3).nOut(4).build(), "input2")
                .addVertex("merge", new MergeVertex(), "L1", "L2")
                .addLayer("out", new OutputLayer.Builder().nIn(4+4).nOut(3).build(), "merge")
                .setOutputs("out")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        val server = new JsonModelServer.Builder<float[], Integer>(net)
                .outputSerializer( new IntSerde())
                .inputDeserializer(new FloatSerde())
                .inferenceAdapter(new InferenceAdapter<float[], Integer>() {
                    @Override
                    public MultiDataSet apply(float[] input) {
                        return new MultiDataSet(Nd4j.create(input, 1, input.length), null);
                    }

                    @Override
                    public Integer apply(INDArray... nnOutput) {
                        return nnOutput[0].argMax().getInt(0);
                    }
                })
                .orderedInputNodes("in")
                .orderedOutputNodes("softmax")
                .port(PORT + 1)
                .inferenceMode(SEQUENTIAL)
                .numWorkers(2)
                .parallelMode(false)
                .build();

        val client = JsonRemoteInference.<float[], Integer>builder()
                .endpointAddress("http://localhost:" + (PORT + 1) + "/v1/serving")
                .outputDeserializer(new IntSerde())
                .inputSerializer( new FloatSerde())
                .build();

        try {
            server.start();
            //client.predict(new float[]{0.0f, 1.0f, 2.0f});
        } catch (Exception e){
            e.printStackTrace();
            throw e;
        } finally {
            server.stop();
        }
    }

    @Test
    public void testCompGraph_1() throws Exception {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Sgd(0.01))
                .graphBuilder()
                .addInputs("input")
                .addLayer("L1", new DenseLayer.Builder().nIn(8).nOut(4).build(), "input")
                .addLayer("out1", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(4).nOut(3).build(), "L1")
                .addLayer("out2", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .nIn(4).nOut(2).build(), "L1")
                .setOutputs("out1","out2")
                .build();

        final ComputationGraph net = new ComputationGraph(conf);
        net.init();

        val server = new JsonModelServer.Builder<float[], Integer>(net)
                .outputSerializer( new IntSerde())
                .inputDeserializer(new FloatSerde())
                .inferenceAdapter(new InferenceAdapter<float[], Integer>() {
                    @Override
                    public MultiDataSet apply(float[] input) {
                        return new MultiDataSet(Nd4j.create(input, 1, input.length), null);
                    }

                    @Override
                    public Integer apply(INDArray... nnOutput) {
                        return nnOutput[0].argMax().getInt(0);
                    }
                })
                .orderedInputNodes("input")
                .orderedOutputNodes("out")
                .port(PORT + 1)
                .inferenceMode(SEQUENTIAL)
                .numWorkers(2)
                .parallelMode(false)
                .build();

        val client = JsonRemoteInference.<float[], Integer>builder()
                .endpointAddress("http://localhost:" + (PORT + 1) + "/v1/serving")
                .outputDeserializer(new IntSerde())
                .inputSerializer( new FloatSerde())
                .build();

        try {
            server.start();
            val result = client.predict(new float[]{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
            assertNotNull(result);
        } catch (Exception e){
            e.printStackTrace();
            throw e;
        } finally {
            server.stop();
        }
    }

    private static class FloatSerde implements JsonSerializer<float[]>, JsonDeserializer<float[]>{
        private final ObjectMapper om = new ObjectMapper();

        @Override
        public float[] deserialize(String json) {
            try {
                return om.readValue(json, FloatHolder.class).getFloats();
            } catch (IOException e){
                throw new RuntimeException(e);
            }
        }

        @Override
        public String serialize(float[] o) {
            try{
                return om.writeValueAsString(new FloatHolder(o));
            } catch (IOException e){
                throw new RuntimeException(e);
            }
        }

        //Use float holder so Jackson does ser/de properly (no "{}" otherwise)
        @AllArgsConstructor @NoArgsConstructor @Data
        private static class FloatHolder {
            private float[] floats;
        }
    }

    private static class IntSerde implements JsonSerializer<Integer>, JsonDeserializer<Integer> {
        private final ObjectMapper om = new ObjectMapper();

        @Override
        public Integer deserialize(String json) {
            try {
                return om.readValue(json, Integer.class);
            } catch (IOException e){
                throw new RuntimeException(e);
            }
        }

        @Override
        public String serialize(Integer o) {
            try{
                return om.writeValueAsString(o);
            } catch (IOException e){
                throw new RuntimeException(e);
            }
        }
    }
}