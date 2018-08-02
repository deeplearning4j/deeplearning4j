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

package org.deeplearning4j.spark.ui;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStorage;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 12/10/2016.
 */
public class TestListeners extends BaseSparkTest {

    @Test
    public void testStatsCollection() {

        JavaSparkContext sc = getContext();
        int nExecutors = numExecutors();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(123)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(100).weightInit(WeightInit.XAVIER)
                                        .activation(Activation.RELU).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).nIn(100).nOut(3)
                                                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
                                                        .build())
                        .pretrain(false).backprop(true).build();



        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();


        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1).batchSizePerWorker(5).averagingFrequency(6)
                        .build();

        SparkDl4jMultiLayer net = new SparkDl4jMultiLayer(sc, conf, tm);
        StatsStorage ss = new MapDBStatsStorage(); //In-memory

        net.setListeners(ss, Collections.singletonList(new StatsListener(null)));

        List<DataSet> list = new IrisDataSetIterator(120, 150).next().asList();
        //120 examples, 4 executors, 30 examples per executor -> 6 updates of size 5 per executor

        JavaRDD<DataSet> rdd = sc.parallelize(list);

        net.fit(rdd);

        List<String> sessions = ss.listSessionIDs();
        System.out.println("Sessions: " + sessions);
        assertEquals(1, sessions.size());

        String sid = sessions.get(0);

        List<String> typeIDs = ss.listTypeIDsForSession(sid);
        List<String> workers = ss.listWorkerIDsForSession(sid);

        System.out.println(sid + "\t" + typeIDs + "\t" + workers);

        List<Persistable> lastUpdates = ss.getLatestUpdateAllWorkers(sid, StatsListener.TYPE_ID);
        System.out.println(lastUpdates);

        System.out.println("Static info:");
        for (String wid : workers) {
            Persistable staticInfo = ss.getStaticInfo(sid, StatsListener.TYPE_ID, wid);
            System.out.println(sid + "\t" + wid);
        }

        assertEquals(1, typeIDs.size());
        assertEquals(numExecutors(), workers.size());
        String firstWorker = workers.get(0);
        String firstWorkerSubstring = workers.get(0).substring(0, firstWorker.length() - 1);
        for (String wid : workers) {
            String widSubstring = wid.substring(0, wid.length() - 1);
            assertEquals(firstWorkerSubstring, widSubstring);

            String counterVal = wid.substring(wid.length() - 1, wid.length());
            int cv = Integer.parseInt(counterVal);
            assertTrue(0 <= cv && cv < numExecutors());
        }
    }
}
