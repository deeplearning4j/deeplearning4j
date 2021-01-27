/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.remote;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.remote.clients.JsonRemoteInference;
import org.nd4j.remote.clients.serde.JsonDeserializer;
import org.nd4j.remote.helpers.House;
import org.nd4j.remote.helpers.HouseToPredictedPriceAdapter;
import org.nd4j.remote.helpers.PredictedPrice;
import org.nd4j.remote.clients.serde.impl.FloatArraySerde;

import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import static org.junit.Assert.*;

@Slf4j
public class SameDiffJsonModelServerTest extends BaseND4JTest {

    @Test
    public void basicServingTest_1() throws Exception {
        val sd = SameDiff.create();
        val sdVariable = sd.placeHolder("input", DataType.INT, 4);
        val result = sdVariable.add(1.0);
        val total = result.mean("total", Integer.MAX_VALUE);

        val server = new SameDiffJsonModelServer.Builder<House, PredictedPrice>()
                .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                .inputDeserializer(new House.HouseDeserializer())
                .inferenceAdapter(new HouseToPredictedPriceAdapter())
                .orderedInputNodes(new String[]{"input"})
                .orderedOutputNodes(new String[]{"total"})
                .sdModel(sd)
                .port(18080)
                .build();

        server.start();

        val client = JsonRemoteInference.<House, PredictedPrice>builder()
                .inputSerializer(new House.HouseSerializer())
                .outputDeserializer(new PredictedPrice.PredictedPriceDeserializer())
                .endpointAddress("http://localhost:18080/v1/serving")
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

        server.stop();
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

    @Test
    public void testDeserialization_3() {
        float[] data = {0.0f,  0.1f, 0.2f};
        val serialized = new FloatArraySerde().serialize(data);
        val deserialized = new FloatArraySerde().deserialize(serialized);
        assertArrayEquals(data, deserialized, 1e-5f);
    }

    @Test(expected = NullPointerException.class)
    public void negativeServingTest_1() throws Exception {
        val sd = SameDiff.create();
        val sdVariable = sd.placeHolder("input", DataType.INT, 4);
        val result = sdVariable.add(1.0);
        val total = result.mean("total", Integer.MAX_VALUE);

        val server = new SameDiffJsonModelServer.Builder<House, PredictedPrice>()
                .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                .inputDeserializer(null)
                .sdModel(sd)
                .port(18080)
                .build();
    }

    @Test(expected = NullPointerException.class)
    public void negativeServingTest_2() throws Exception {
        val sd = SameDiff.create();
        val sdVariable = sd.placeHolder("input", DataType.INT, 4);
        val result = sdVariable.add(1.0);
        val total = result.mean("total", Integer.MAX_VALUE);

        val server = new SameDiffJsonModelServer.Builder<House, PredictedPrice>()
                .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                .inputDeserializer(new House.HouseDeserializer())
                .inferenceAdapter(new HouseToPredictedPriceAdapter())
                .sdModel(sd)
                .port(18080)
                .build();

    }

    @Test(expected = IOException.class)
    public void negativeServingTest_3() throws Exception {
        val sd = SameDiff.create();
        val sdVariable = sd.placeHolder("input", DataType.INT, 4);
        val result = sdVariable.add(1.0);
        val total = result.mean("total", Integer.MAX_VALUE);

        val server = new SameDiffJsonModelServer.Builder<House, PredictedPrice>()
                .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                .inputDeserializer(new House.HouseDeserializer())
                .inferenceAdapter(new HouseToPredictedPriceAdapter())
                .orderedInputNodes(new String[]{"input"})
                .orderedOutputNodes(new String[]{"total"})
                .sdModel(sd)
                .port(18080)
                .build();

        server.start();

        val client = JsonRemoteInference.<House, PredictedPrice>builder()
                .inputSerializer(new House.HouseSerializer())
                .outputDeserializer(new JsonDeserializer<PredictedPrice>() {
                    @Override
                    public PredictedPrice deserialize(String json) {
                        return null;
                    }
                })
                .endpointAddress("http://localhost:18080/v1/serving")
                .build();

        int district = 2;
        House house = House.builder().area(100).bathrooms(2).bedrooms(3).district(district).build();

        // warmup
        PredictedPrice price = client.predict(house);

        server.stop();
    }

    @Test
    public void asyncServingTest() throws Exception {
        val sd = SameDiff.create();
        val sdVariable = sd.placeHolder("input", DataType.INT, 4);
        val result = sdVariable.add(1.0);
        val total = result.mean("total", Integer.MAX_VALUE);

        val server = new SameDiffJsonModelServer.Builder<House, PredictedPrice>()
                .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                .inputDeserializer(new House.HouseDeserializer())
                .inferenceAdapter(new HouseToPredictedPriceAdapter())
                .orderedInputNodes(new String[]{"input"})
                .orderedOutputNodes(new String[]{"total"})
                .sdModel(sd)
                .port(18080)
                .build();

        server.start();

        val client = JsonRemoteInference.<House, PredictedPrice>builder()
                .inputSerializer(new House.HouseSerializer())
                .outputDeserializer(new PredictedPrice.PredictedPriceDeserializer())
                .endpointAddress("http://localhost:18080/v1/serving")
                .build();

        int district = 2;
        House house = House.builder().area(100).bathrooms(2).bedrooms(3).district(district).build();

        val timeStart = System.currentTimeMillis();
        Future<PredictedPrice> price = client.predictAsync(house);
        assertNotNull(price);
        assertEquals((float) district + 1.0f, price.get().getPrice(), 1e-5);
        val timeStop = System.currentTimeMillis();

        log.info("Time spent: {} ms", timeStop - timeStart);


        server.stop();
    }

    @Test
    public void negativeAsyncTest() throws Exception {
        val sd = SameDiff.create();
        val sdVariable = sd.placeHolder("input", DataType.INT, 4);
        val result = sdVariable.add(1.0);
        val total = result.mean("total", Integer.MAX_VALUE);

        val server = new SameDiffJsonModelServer.Builder<House, PredictedPrice>()
                .outputSerializer(new PredictedPrice.PredictedPriceSerializer())
                .inputDeserializer(new House.HouseDeserializer())
                .inferenceAdapter(new HouseToPredictedPriceAdapter())
                .orderedInputNodes(new String[]{"input"})
                .orderedOutputNodes(new String[]{"total"})
                .sdModel(sd)
                .port(18080)
                .build();

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
                .endpointAddress("http://localhost:18080/v1/serving")
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

        server.stop();
    }

}