/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j;

import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.common.resources.Resources;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.nio.file.Files;
import java.util.concurrent.CountDownLatch;

@Ignore
public class RandomTests extends BaseDL4JTest {

    @Test
    public void testReproduce() throws Exception {

        final MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new RmsProp())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(28 * 28).nOut(10)
                        .activation(Activation.TANH).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                        LossFunctions.LossFunction.MCXENT).nIn(10).nOut(10)
                        .activation(Activation.SOFTMAX).build())
                .build();

        for (int e = 0; e < 3; e++) {

            int nThreads = 10;
            final CountDownLatch l = new CountDownLatch(nThreads);
            for (int i = 0; i < nThreads; i++) {
                final int j = i;
                Thread t = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            MultiLayerNetwork net = new MultiLayerNetwork(conf.clone());
                            net.init();
                            DataSetIterator iter = new EarlyTerminationDataSetIterator(new MnistDataSetIterator(10, false, 12345), 100);
                            net.fit(iter);
                        } catch (Throwable t) {
                            System.out.println("Thread failed: " + j);
                            t.printStackTrace();
                        } finally {
                            l.countDown();
                        }
                    }
                });
                t.start();
            }

            l.await();
            System.out.println("DONE " + e + "\n");
        }
    }
}
