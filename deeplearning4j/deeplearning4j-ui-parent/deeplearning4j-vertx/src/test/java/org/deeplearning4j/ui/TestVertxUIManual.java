/*
 *  ******************************************************************************
 *  *
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

package org.deeplearning4j.ui;

import io.netty.handler.codec.http.HttpResponseStatus;
import io.vertx.core.Future;
import io.vertx.core.Promise;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.common.function.Function;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.UnsupportedEncodingException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.HashMap;
import java.util.concurrent.CountDownLatch;

import static org.junit.Assert.*;

@Slf4j
@Ignore
public class TestVertxUIManual extends BaseDL4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        return 3600_000L;
    }

    @Test
    @Ignore
    public void testUI() throws Exception {
        VertxUIServer uiServer = (VertxUIServer) UIServer.getInstance();
        assertEquals(9000, uiServer.getPort());

        Thread.sleep(3000_000);
        uiServer.stop();
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

            for (int i = 0; i < 100; i++) {
                net.fit(iter);
            }
            Thread.sleep(5_000);
        }
    }

    @Test
    @Ignore
    public void testUIServerStop() throws Exception {
        UIServer uiServer = UIServer.getInstance(true, null);
        assertTrue(uiServer.isMultiSession());
        assertFalse(uiServer.isStopped());

        long sleepMilliseconds = 30_000;
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
    @Ignore
    public void testUIServerStopAsync() throws Exception {
        UIServer uiServer = UIServer.getInstance(true, null);
        assertTrue(uiServer.isMultiSession());
        assertFalse(uiServer.isStopped());

        long sleepMilliseconds = 30_000;
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

    @Test
    @Ignore
    public void testUIAutoAttachDetach() throws Exception {
        long detachTimeoutMillis = 15_000;
        AutoDetachingStatsStorageProvider statsProvider = new AutoDetachingStatsStorageProvider(detachTimeoutMillis);
        UIServer uIServer = UIServer.getInstance(true, statsProvider);
        statsProvider.setUIServer(uIServer);
        InMemoryStatsStorage ss = null;
        for (int session = 0; session < 3; session++) {
            int layerSize = session + 4;

            ss = new InMemoryStatsStorage();
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

        Thread.sleep(detachTimeoutMillis + 60_000);
        assertFalse(uIServer.isAttached(ss));
    }


    /**
     * Get URL-encoded URL for training session on given server address
     * @param serverAddress server address
     * @param sessionId session ID
     * @return URL
     * @throws UnsupportedEncodingException if the used encoding is not supported
     */
    private static String trainingSessionUrl(String serverAddress, String sessionId)
            throws UnsupportedEncodingException {
        return String.format("%s/train/%s", serverAddress, URLEncoder.encode(sessionId, "UTF-8"));
    }

    /**
     * StatsStorage provider with automatic detaching of StatsStorage after a timeout
     * @author Tamas Fenyvesi
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
                        log.info("Waiting to detach StatsStorage (session ID: {})" +
                                " after {} ms ", sessionId, autoDetachTimeoutMillis);
                        Thread.sleep(autoDetachTimeoutMillis);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    } finally {
                        log.info("Auto-detaching StatsStorage (session ID: {}) after {} ms.",
                                sessionId, autoDetachTimeoutMillis);
                        uIServer.detach(statsStorage);
                        log.info(" To re-attach StatsStorage of training session, visit {}/train/{}",
                                uIServer.getAddress(), sessionId);
                    }
                }).start();
            }

            return statsStorage;
        }
    }

}
