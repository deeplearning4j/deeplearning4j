/* ******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

import io.netty.handler.codec.http.HttpResponseStatus;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
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
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.HashMap;

import static org.junit.Assert.*;

/**
 * @author Tamas Fenyvesi
 */
@Ignore
@Slf4j
public class TestVertxUIMultiSession extends BaseDL4JTest {
    @Before
    public void setUp() throws Exception {
        UIServer.stopInstance();
    }

    @Test
    public void testUIMultiSession() throws Exception {

        UIServer uIServer = UIServer.getInstance(true, null);
        HashMap<Thread, StatsStorage> statStorageForThread = new HashMap<>();
        HashMap<Thread, String> sessionIdForThread = new HashMap<>();

        for (int session = 0; session < 10; session++) {

            StatsStorage ss = new InMemoryStatsStorage();

            final int sid = session;
            final String sessionId = Integer.toString(sid);

            Thread training = new Thread(() -> {
                int layerSize = sid + 4;
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .updater(new Adam(1e-2))
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                        .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(4).nOut(layerSize).build())
                        .layer(1, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX).nIn(layerSize).nOut(3).build())
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                StatsListener statsListener = new StatsListener(ss, 1);

                statsListener.setSessionID(sessionId);
                net.setListeners(statsListener, new ScoreIterationListener(1));
                uIServer.attach(ss);

                DataSetIterator iter = new IrisDataSetIterator(150, 150);

                for (int i = 0; i < 20; i++) {
                    net.fit(iter);
                }
            });

            training.start();
            statStorageForThread.put(training, ss);
            sessionIdForThread.put(training, sessionId);
        }

        Thread.sleep(10000000);

        for (Thread thread: statStorageForThread.keySet()) {
            StatsStorage ss = statStorageForThread.get(thread);
            String sessionId = sessionIdForThread.get(thread);
            try {
                thread.join();
                /*
                 * Visiting /train/:sessionId to check if training session is available on it's URL
                 */
                String sessionUrl = trainingSessionUrl(uIServer.getAddress(), sessionId);
                HttpURLConnection conn = (HttpURLConnection) new URL(sessionUrl).openConnection();
                conn.connect();

                assertEquals(HttpResponseStatus.OK.code(), conn.getResponseCode());
                assertTrue(uIServer.isAttached(ss));
            } catch (InterruptedException | IOException e) {
                log.error("",e);
                fail(e.getMessage());
            } finally {
                uIServer.detach(ss);
                assertFalse(uIServer.isAttached(ss));
            }
        }

    }

    @Test
    public void testUIAutoAttach() throws Exception {
        HashMap<String, StatsStorage> statsStorageForSession = new HashMap<>();

        Function<String, StatsStorage> statsStorageProvider = statsStorageForSession::get;
        UIServer uIServer = UIServer.getInstance(true, statsStorageProvider);

        for (int session = 0; session < 3; session++) {
            int layerSize = session + 4;

            InMemoryStatsStorage ss = new InMemoryStatsStorage();
            String sessionId = Integer.toString(session);
            statsStorageForSession.put(sessionId, ss);
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                    .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(4).nOut(layerSize).build())
                    .layer(1, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                            .activation(Activation.SOFTMAX).nIn(layerSize).nOut(3).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            StatsListener statsListener = new StatsListener(ss, 1);
            statsListener.setSessionID(sessionId);
            net.setListeners(statsListener, new ScoreIterationListener(1));
            uIServer.attach(ss);

            DataSetIterator iter = new IrisDataSetIterator(150, 150);

            for (int i = 0; i < 20; i++) {
                net.fit(iter);
            }

            assertTrue(uIServer.isAttached(statsStorageForSession.get(sessionId)));
            uIServer.detach(ss);
            assertFalse(uIServer.isAttached(statsStorageForSession.get(sessionId)));

            /*
             * Visiting /train/:sessionId to auto-attach StatsStorage
             */
            String sessionUrl = trainingSessionUrl(uIServer.getAddress(), sessionId);
            HttpURLConnection conn = (HttpURLConnection) new URL(sessionUrl).openConnection();
            conn.connect();

            assertEquals(HttpResponseStatus.OK.code(), conn.getResponseCode());
            assertTrue(uIServer.isAttached(statsStorageForSession.get(sessionId)));
        }
    }

    @Test
    @Ignore
    public void testUIAutoAttachDetach() throws Exception {

        long autoDetachTimeoutMillis = 30_000;
        AutoDetachingStatsStorageProvider statsProvider = new AutoDetachingStatsStorageProvider(autoDetachTimeoutMillis);
        UIServer uIServer = UIServer.getInstance(true, statsProvider);
        statsProvider.setUIServer(uIServer);

        for (int session = 0; session < 3; session++) {
            int layerSize = session + 4;

            InMemoryStatsStorage ss = new InMemoryStatsStorage();
            String sessionId = Integer.toString(session);
            statsProvider.put(sessionId, ss);
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                    .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(4).nOut(layerSize).build())
                    .layer(1, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                            .activation(Activation.SOFTMAX).nIn(layerSize).nOut(3).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            StatsListener statsListener = new StatsListener(ss, 1);
            statsListener.setSessionID(sessionId);
            net.setListeners(statsListener, new ScoreIterationListener(1));
            uIServer.attach(ss);

            DataSetIterator iter = new IrisDataSetIterator(150, 150);

            for (int i = 0; i < 20; i++) {
                net.fit(iter);
            }

            assertTrue(uIServer.isAttached(ss));
            uIServer.detach(ss);
            assertFalse(uIServer.isAttached(ss));

            /*
             * Visiting /train/:sessionId to auto-attach StatsStorage
             */
            String sessionUrl = trainingSessionUrl(uIServer.getAddress(), sessionId);
            HttpURLConnection conn = (HttpURLConnection) new URL(sessionUrl).openConnection();
            conn.connect();

            assertEquals(HttpResponseStatus.OK.code(), conn.getResponseCode());
            assertTrue(uIServer.isAttached(ss));
        }

        Thread.sleep(1_000_000);
    }

    @Test (expected = RuntimeException.class)
    public void testUIServerGetInstanceMultipleCalls1() {
        UIServer uiServer = UIServer.getInstance();
        assertFalse(uiServer.isMultiSession());
        UIServer.getInstance(true, null);

    }

    @Test (expected = RuntimeException.class)
    public void testUIServerGetInstanceMultipleCalls2() {
        UIServer uiServer = UIServer.getInstance(true, null);
        assertTrue(uiServer.isMultiSession());
        UIServer.getInstance(false, null);
    }

    /**
     * Get URL-encoded URL for training session on given server address
     * @param serverAddress server address
     * @param sessionId session ID
     * @return URL
     * @throws UnsupportedEncodingException if the used encoding is not supported
     */
    private static String trainingSessionUrl(String serverAddress, String sessionId) throws UnsupportedEncodingException {
        return String.format("%s/train/%s", serverAddress, URLEncoder.encode(sessionId, "UTF-8"));
    }

    /**
     * StatsStorage provider with automatic detaching of StatsStorage after a timeout
     * @author fenyvesit
     *
     */
    private static class AutoDetachingStatsStorageProvider implements Function<String, StatsStorage> {
        HashMap<String, InMemoryStatsStorage> storageForSession = new HashMap<>();
        UIServer uIServer;
        long autoDetachTimeoutMillis;

        public AutoDetachingStatsStorageProvider(long autoDetachTimeoutMillis) {
            this.autoDetachTimeoutMillis = autoDetachTimeoutMillis;
        }

        public void put(String sessionId, InMemoryStatsStorage statsStorage) {
            storageForSession.put(sessionId, statsStorage);
        }

        public void setUIServer(UIServer uIServer) {
            this.uIServer = uIServer;
        }

        @Override
        public StatsStorage apply(String sessionId) {
            StatsStorage statsStorage = storageForSession.get(sessionId);

            if (statsStorage != null) {
                new Thread(() -> {
                    try {
                        System.out.println("Waiting to detach StatsStorage (session ID: " + sessionId + ")" +
                                " after " + autoDetachTimeoutMillis + " ms ");
                        Thread.sleep(autoDetachTimeoutMillis);
                    } catch (InterruptedException e) {
                        log.error("",e);
                    } finally {
                        System.out.println("Auto-detaching StatsStorage (session ID: " + sessionId + ") after " +
                                autoDetachTimeoutMillis + " ms.");
                        uIServer.detach(statsStorage);
                        System.out.println(" To re-attach StatsStorage of training session, visit " +
                                uIServer.getAddress() + "/train/" + sessionId);
                    }
                }).start();
            }

            return statsStorage;
        }
    }
}
