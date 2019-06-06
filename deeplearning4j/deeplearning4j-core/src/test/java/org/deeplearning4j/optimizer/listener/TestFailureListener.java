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

package org.deeplearning4j.optimizer.listener;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.FailureTestingListener;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.net.InetAddress;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;

/**
 * WARNING: DO NOT ENABLE (UN-IGNORE) THESE TESTS.
 * They should be run manually, not as part of standard unit test run.
 */
@Ignore
public class TestFailureListener extends BaseDL4JTest {

    @Ignore
    @Test
    public void testFailureIter5() throws Exception {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-4))
                .list()
                .layer(0, new OutputLayer.Builder().nIn(4).nOut(3).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new FailureTestingListener(
//                FailureTestingListener.FailureMode.OOM,
                FailureTestingListener.FailureMode.SYSTEM_EXIT_1,
                new FailureTestingListener.IterationEpochTrigger(false, 10)));

        DataSetIterator iter = new IrisDataSetIterator(5,150);

        net.fit(iter);
    }

    @Ignore
    @Test
    public void testFailureRandom_OR(){

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-4))
                .list()
                .layer(0, new OutputLayer.Builder().nIn(4).nOut(3).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        String username = System.getProperty("user.name");
        assertNotNull(username);
        assertFalse(username.isEmpty());

        net.setListeners(new FailureTestingListener(
                FailureTestingListener.FailureMode.SYSTEM_EXIT_1,
                new FailureTestingListener.Or(
                        new FailureTestingListener.IterationEpochTrigger(false, 10000),
                        new FailureTestingListener.RandomProb(FailureTestingListener.CallType.ANY, 0.02))
                ));

        DataSetIterator iter = new IrisDataSetIterator(5,150);

        net.fit(iter);
    }

    @Ignore
    @Test
    public void testFailureRandom_AND() throws Exception {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-4))
                .list()
                .layer(0, new OutputLayer.Builder().nIn(4).nOut(3).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        String hostname = InetAddress.getLocalHost().getHostName();
        assertNotNull(hostname);
        assertFalse(hostname.isEmpty());

        net.setListeners(new FailureTestingListener(
                FailureTestingListener.FailureMode.ILLEGAL_STATE,
                new FailureTestingListener.And(
                        new FailureTestingListener.HostNameTrigger(hostname),
                        new FailureTestingListener.RandomProb(FailureTestingListener.CallType.ANY, 0.05))
        ));

        DataSetIterator iter = new IrisDataSetIterator(5,150);

        net.fit(iter);
    }

}
