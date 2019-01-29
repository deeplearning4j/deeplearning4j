/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.ui.play;

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
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.HashMap;

import static org.junit.Assert.*;

/**
 * @author Tamas Fenyvesi
 */
@Ignore
public class TestPlayUIMultiSession {

    @Test
    @Ignore
    public void testUIMultiSession() throws Exception {

        UIServer uiServer = UIServer.getInstance(true, null);

        for (int session = 0; session < 3; session++) {

            StatsStorage ss = new InMemoryStatsStorage();

            final int sid = session;
            new Thread(() -> {
                int layerSize = sid + 4;
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                        .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(4).nOut(layerSize).build())
                        .layer(1, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX).nIn(layerSize).nOut(3).build())
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                StatsListener statsListener = new StatsListener(ss);
                String sessionId = Integer.toString(sid);
                statsListener.setSessionID(sessionId);
                net.setListeners(statsListener, new ScoreIterationListener(1));
                uiServer.attach(ss);

                DataSetIterator iter = new IrisDataSetIterator(150, 150);

                for (int i = 0; i < 20; i++) {
                    net.fit(iter);
                }
                try {
                    Thread.sleep(600_000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                    fail(e.getMessage());
                } finally {
                    uiServer.detach(ss);
                }
            }).start();
        }

        Thread.sleep(1_000_000);
    }

    @Test
    @Ignore
    public void testUIStatsStorageProvider() throws Exception {

        AutoDetachingStatsStorageProvider statsProvider = new AutoDetachingStatsStorageProvider();
        UIServer playUIServer = UIServer.getInstance(true, statsProvider);
        statsProvider.setUIServer(playUIServer);

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
            playUIServer.attach(ss);

            DataSetIterator iter = new IrisDataSetIterator(150, 150);

            for (int i = 0; i < 20; i++) {
                net.fit(iter);
            }

            /*
             * Wait for the first update (containing session ID) to effectively attach StatsStorage in PlayUIServer.
            */
            Thread.sleep(1000);

            playUIServer.detach(ss);
            System.out.println("To re-attach StatsStorage of training session, visit "
                    + playUIServer.getAddress() + "/train/" + sessionId);
        }

        Thread.sleep(1_000_000);
    }

    /**
     * StatsStorage provider with automatic detaching of StatsStorage after a timeout
     * @author fenyvesit
     *
     */
    private static class AutoDetachingStatsStorageProvider implements Function<String, StatsStorage> {

        HashMap<String, InMemoryStatsStorage> storageForSession = new HashMap<>();
        UIServer uIServer;

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
                // auto-detach StatsStorage instances that will be attached via this provider
                long autoDetachTimeoutMillis = 1000*30;
                new Thread(() -> {
                    try {
                        System.out.println("Waiting to detach StatsStorage (session ID: " + sessionId + ")" +
                                " after " + autoDetachTimeoutMillis + " ms ");
                        Thread.sleep(autoDetachTimeoutMillis);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    } finally {
                        System.out.println("Auto-detaching StatsStorage (session ID:" + sessionId + ") after " +
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
