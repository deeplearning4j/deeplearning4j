/*-
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.deeplearning4j.spark.impl.customlayer;

import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.impl.customlayer.layer.CustomLayer;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by Alex on 28/08/2016.
 */
public class TestCustomLayer extends BaseSparkTest {

    @Test
    public void testSparkWithCustomLayer() {
        //Basic test - checks whether exceptions etc are thrown with custom layers + spark
        //Custom layers are tested more extensively in dl4j core
        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder().updater(new Sgd(0.1)).list()
                                        .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build())
                                        .layer(1, new CustomLayer(3.14159)).layer(2,
                                                        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                                        .nIn(10).nOut(10).build())
                                        .pretrain(false).backprop(true).build();

        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1).averagingFrequency(2)
                        .batchSizePerWorker(5).saveUpdater(true).workerPrefetchNumBatches(0).build();

        SparkDl4jMultiLayer net = new SparkDl4jMultiLayer(sc, conf, tm);

        List<DataSet> testData = new ArrayList<>();
        Random r = new Random(12345);
        for (int i = 0; i < 200; i++) {
            INDArray f = Nd4j.rand(1, 10);
            INDArray l = Nd4j.zeros(1, 10);
            l.putScalar(0, r.nextInt(10), 1.0);
            testData.add(new DataSet(f, l));
        }

        JavaRDD<DataSet> rdd = sc.parallelize(testData);
        net.fit(rdd);
    }

}
