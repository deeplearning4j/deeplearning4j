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

package org.deeplearning4j.parallelism;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

@Slf4j
public class ThreadSafetyTests {
    private ComputationGraphConfiguration cgConf = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .graphBuilder()
            .addInputs("alpha")
            .addLayer("beta", new DenseLayer.Builder()
                    .activation(Activation.RELU)
                    .nIn(10)
                    .nOut(10)
                    .build(), "alpha")
            .addLayer("omega", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .activation(Activation.SOFTMAX)
                    .nIn(10)
                    .nOut(10)
                    .build(), "beta")
            .setOutputs("omega")
            .build();

    private MultiLayerConfiguration mlnConf = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new DenseLayer.Builder()
                    .activation(Activation.RELU)
                    .nIn(10)
                    .nOut(10)
                    .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .activation(Activation.SOFTMAX)
                    .nIn(10)
                    .nOut(10)
                    .build())
            .build();

    @Test (timeout = 5000L)
    public void testSequentialAccess_1() throws Exception {
        val e = new AtomicBoolean(false);
        final val model = new ComputationGraph(cgConf);
        model.init();

        val t = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    val array1 = Nd4j.create(10, 10);
                    model.output(array1);

                    val array2 = Nd4j.create(10, 10);
                    model.output(array2);
                } catch (Exception e1) {
                    e.set(true);
                }
            }
        });

        t.start();
        t.join();

        assertFalse(e.get());
    }

    @Test (timeout = 5000L)
    public void testSequentialAccess_2() throws Exception {
        val e = new AtomicBoolean(false);
        final val model = new MultiLayerNetwork(mlnConf);
        model.init();

        val t = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    val array1 = Nd4j.create(10, 10);
                    model.output(array1);

                    val array2 = Nd4j.create(10, 10);
                    model.output(array2);
                } catch (Exception e1) {
                    e.set(true);
                }
            }
        });

        t.start();
        t.join();

        assertFalse(e.get());
    }

    @Test(expected = ND4JIllegalStateException.class, timeout = 5000L)
    public void testConcurrentAccess_1() throws Exception {
        val e = new AtomicBoolean(false);
        final val model = new MultiLayerNetwork(mlnConf);
        model.init();

        val t1 = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    val array = Nd4j.create(10, 10);
                    model.output(array);
                } catch (ND4JIllegalStateException e1) {
                    e.set(true);
                }
            }
        });

        val t2 = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    val array = Nd4j.create(10, 10);
                    model.output(array);
                } catch (ND4JIllegalStateException e1) {
                    e.set(true);
                }
            }
        });

        t1.start();
        t2.start();

        t1.join();
        t2.join();

        assertTrue(e.get());
    }
}
