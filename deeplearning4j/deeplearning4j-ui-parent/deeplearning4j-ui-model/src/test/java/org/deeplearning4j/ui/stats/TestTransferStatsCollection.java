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

package org.deeplearning4j.ui.stats;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;

import java.io.File;
import java.io.IOException;

/**
 * Created by Alex on 07/04/2017.
 */
public class TestTransferStatsCollection extends BaseDL4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        return 90_000L;
    }

    @Test
    public void test() throws IOException {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build())
                        .layer(1, new OutputLayer.Builder().activation(Activation.SOFTMAX).nIn(10).nOut(10).build()).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();


        MultiLayerNetwork net2 =
                        new TransferLearning.Builder(net)
                                        .fineTuneConfiguration(
                                                        new FineTuneConfiguration.Builder().updater(new Sgd(0.01)).build())
                                        .setFeatureExtractor(0).build();

        net2.setListeners(new StatsListener(new InMemoryStatsStorage()));

        //Previosuly: failed on frozen layers
        net2.fit(new DataSet(Nd4j.rand(8, 10), Nd4j.rand(8, 10)));
    }
}
