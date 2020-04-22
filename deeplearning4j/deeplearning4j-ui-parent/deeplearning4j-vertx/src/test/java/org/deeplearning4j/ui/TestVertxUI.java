/* ******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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

package org.deeplearning4j.ui;

import io.vertx.core.Future;
import io.vertx.core.Promise;
import io.vertx.core.Vertx;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.exception.DL4JException;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.conf.serde.JsonMappers;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.*;

/**
 * Created by Alex on 08/10/2016.
 */
@Slf4j
@Ignore
public class TestVertxUI extends BaseDL4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        return 600_000L;
    }

    @Before
    public void setUp() throws Exception {
        UIServer.stopInstance();
    }

    @Test
    @Ignore
    public void testUI() throws Exception {
        VertxUIServer uiServer = (VertxUIServer) UIServer.getInstance();
        assertEquals(9000, uiServer.getPort());

        Thread.sleep(30_000);
        uiServer.stop();
    }

    @Test
    @Ignore
    public void testUI_VAE() throws Exception {
        //Variational autoencoder - for unsupervised layerwise pretraining

        StatsStorage ss = new InMemoryStatsStorage();

        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(ss);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Sgd(1e-5))
                        .list().layer(0,
                                        new VariationalAutoencoder.Builder().nIn(4).nOut(3).encoderLayerSizes(10, 11)
                                                        .decoderLayerSizes(12, 13).weightInit(WeightInit.XAVIER)
                                                        .pzxActivationFunction(Activation.IDENTITY)
                                                        .reconstructionDistribution(
                                                                        new GaussianReconstructionDistribution())
                                                        .activation(Activation.LEAKYRELU).build())
                        .layer(1, new VariationalAutoencoder.Builder().nIn(3).nOut(3).encoderLayerSizes(7)
                                        .decoderLayerSizes(8).weightInit(WeightInit.XAVIER)
                                        .pzxActivationFunction(Activation.IDENTITY)
                                        .reconstructionDistribution(new GaussianReconstructionDistribution())
                                        .activation(Activation.LEAKYRELU).build())
                        .layer(2, new OutputLayer.Builder().nIn(3).nOut(3).build())
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new StatsListener(ss), new ScoreIterationListener(1));

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        for (int i = 0; i < 50; i++) {
            net.fit(iter);
            Thread.sleep(100);
        }


        Thread.sleep(100000);
    }


    @Test
    @Ignore
    public void testUIMultipleSessions() throws Exception {

        for (int session = 0; session < 3; session++) {

            StatsStorage ss = new InMemoryStatsStorage();

            UIServer uiServer = UIServer.getInstance();
            uiServer.attach(ss);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                    .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(4).nOut(4).build())
                    .layer(1, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                            .activation(Activation.SOFTMAX).nIn(4).nOut(3).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            net.setListeners(new StatsListener(ss, 1), new ScoreIterationListener(1));

            DataSetIterator iter = new IrisDataSetIterator(150, 150);

            for (int i = 0; i < 20; i++) {
                net.fit(iter);
                Thread.sleep(100);
            }
        }


        Thread.sleep(1000000);
    }
    
    @Test
    @Ignore
    public void testUISequentialSessions() throws Exception {
        UIServer uiServer = UIServer.getInstance();
        StatsStorage ss = null;
        for (int session = 0; session < 3; session++) {
            
            if (ss != null) {
                uiServer.detach(ss);
            }
            ss = new InMemoryStatsStorage();
            uiServer.attach(ss);

            int numInputs = 4;
            int outputNum = 3;
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.03))
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(outputNum).build())
                .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            net.setListeners(new StatsListener(ss), new ScoreIterationListener(1));

            DataSetIterator iter = new IrisDataSetIterator(150, 150);

            for (int i = 0; i < 1000; i++) {
                net.fit(iter);
            }
            Thread.sleep(5000);
        }

        Thread.sleep(1000000);
    }

    @Test
    @Ignore
    public void testUICompGraph() throws Exception {

        StatsStorage ss = new InMemoryStatsStorage();

        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(ss);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                        .addLayer("L0", new DenseLayer.Builder().activation(Activation.TANH).nIn(4).nOut(4).build(),
                                        "in")
                        .addLayer("L1", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nIn(4).nOut(3).build(), "L0")
                        .setOutputs("L1").build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        net.setListeners(new StatsListener(ss), new ScoreIterationListener(1));

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        for (int i = 0; i < 100; i++) {
            net.fit(iter);
            Thread.sleep(100);
        }

        Thread.sleep(1000000);
    }

    @Test
    public void testAutoAttach() throws Exception {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                .addLayer("L0", new DenseLayer.Builder().activation(Activation.TANH).nIn(4).nOut(4).build(),
                        "in")
                .addLayer("L1", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(4).nOut(3).build(), "L0")
                .setOutputs("L1").build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        StatsStorage ss1 = new InMemoryStatsStorage();

        net.setListeners(new StatsListener(ss1, 1, "ss1"));

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        for (int i = 0; i < 5; i++) {
            net.fit(iter);
        }

        StatsStorage ss2 = new InMemoryStatsStorage();
        net.setListeners(new StatsListener(ss2, 1, "ss2"));

        for (int i = 0; i < 4; i++) {
            net.fit(iter);
        }

        UIServer ui = UIServer.getInstance(true, null);
        try {
            ((VertxUIServer) ui).autoAttachStatsStorageBySessionId(new Function<String, StatsStorage>() {
                @Override
                public StatsStorage apply(String s) {
                    if ("ss1".equals(s)) {
                        return ss1;
                    } else if ("ss2".equals(s)) {
                        return ss2;
                    }
                    return null;
                }
            });

            String json1 = IOUtils.toString(new URL("http://localhost:9000/train/ss1/overview/data"),
                    StandardCharsets.UTF_8);

            String json2 = IOUtils.toString(new URL("http://localhost:9000/train/ss2/overview/data"),
                    StandardCharsets.UTF_8);

            assertNotEquals(json1, json2);

            Map<String, Object> m1 = JsonMappers.getMapper().readValue(json1, Map.class);
            Map<String, Object> m2 = JsonMappers.getMapper().readValue(json2, Map.class);

            List<Object> s1 = (List<Object>) m1.get("scores");
            List<Object> s2 = (List<Object>) m2.get("scores");
            assertEquals(5, s1.size());
            assertEquals(4, s2.size());
        } finally {
            ui.stop();
        }
    }

    @Test
    public void testUIAttachDetach() throws Exception {
        StatsStorage ss = new InMemoryStatsStorage();

        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(ss);
        assertFalse(uiServer.getStatsStorageInstances().isEmpty());
        uiServer.detach(ss);
        assertTrue(uiServer.getStatsStorageInstances().isEmpty());
    }

    @Test
    public void testUIServerStop() throws Exception {
        UIServer uiServer = UIServer.getInstance(true, null);
        assertTrue(uiServer.isMultiSession());
        assertFalse(uiServer.isStopped());

        long sleepMilliseconds = 10_000;
        log.info("Waiting {} ms before stopping.", sleepMilliseconds);
        Thread.sleep(sleepMilliseconds);
        uiServer.stop();
        assertTrue(uiServer.isStopped());

        log.info("UI server is stopped. Waiting {} ms before starting new UI server.", sleepMilliseconds);
        Thread.sleep(sleepMilliseconds);
        uiServer = UIServer.getInstance(false, null);
        assertFalse(uiServer.isMultiSession());
        assertFalse(uiServer.isStopped());

        log.info("Waiting {} ms before stopping.", sleepMilliseconds);
        Thread.sleep(sleepMilliseconds);
        uiServer.stop();
        assertTrue(uiServer.isStopped());
    }


    @Test
    public void testUIServerStopAsync() throws Exception {
        UIServer uiServer = UIServer.getInstance(true, null);
        assertTrue(uiServer.isMultiSession());
        assertFalse(uiServer.isStopped());

        long sleepMilliseconds = 10_000;
        log.info("Waiting {} ms before stopping.", sleepMilliseconds);
        Thread.sleep(sleepMilliseconds);

        CountDownLatch latch = new CountDownLatch(1);
        Promise<Void> promise = Promise.promise();
        promise.future().compose(
                success -> Future.future(prom -> latch.countDown()),
                failure -> Future.future(prom -> latch.countDown())
        );

        uiServer.stopAsync(promise);
        latch.await();
        assertTrue(uiServer.isStopped());

        log.info("UI server is stopped. Waiting {} ms before starting new UI server.", sleepMilliseconds);
        Thread.sleep(sleepMilliseconds);
        uiServer = UIServer.getInstance(false, null);
        assertFalse(uiServer.isMultiSession());

        log.info("Waiting {} ms before stopping.", sleepMilliseconds);
        Thread.sleep(sleepMilliseconds);
        uiServer.stop();
    }

    @Test (expected = DL4JException.class)
    public void testUIStartPortAlreadyBound() throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(1);
        //Create HttpServer that binds the same port
        int port = VertxUIServer.DEFAULT_UI_PORT;
        Vertx vertx = Vertx.vertx();
        vertx.createHttpServer()
                .requestHandler(event -> {})
                .listen(port, result -> latch.countDown());
        latch.await();

        try {
            //DL4JException signals that the port cannot be bound, UI server cannot start
            UIServer.getInstance();
        } finally {
            vertx.close();
        }
    }

    @Test
    public void testUIStartAsync() throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(1);
        Promise<String> promise = Promise.promise();
        promise.future().compose(
                success -> Future.future(prom -> latch.countDown()),
                failure -> Future.future(prom -> latch.countDown())
        );
        int port = VertxUIServer.DEFAULT_UI_PORT;
        VertxUIServer.getInstance(port, false, null, promise);
        latch.await();
        if (promise.future().succeeded()) {
            String deploymentId = promise.future().result();
            log.debug("UI server deployed, deployment ID = {}", deploymentId);
        } else {
            log.debug("UI server failed to deploy.", promise.future().cause());
        }
    }

    @Test
    public void testUIAutoStopOnThreadExit() throws InterruptedException {
        AtomicReference<UIServer> uiServer = new AtomicReference<>();
        Thread thread = new Thread(() -> uiServer.set(UIServer.getInstance()));
        thread.start();
        thread.join();
        Thread.sleep(1_000);
        assertTrue(uiServer.get().isStopped());
    }
}
