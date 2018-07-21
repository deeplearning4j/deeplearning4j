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

package org.deeplearning4j.spark.impl.multilayer;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertTrue;

@Slf4j
public class TestSparkDl4jMultiLayer extends BaseSparkTest {

    @Test
    public void testEvaluationSimple() throws Exception {

        for( int evalWorkers : new int[]{1, 4, 8}) {
            //Simple test to validate DL4J issue 4099 is fixed...

            int numEpochs = 1;
            int batchSizePerWorker = 8;

            //Load the data into memory then parallelize
            //This isn't a good approach in general - but is simple to use for this example
            DataSetIterator iterTrain = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
            DataSetIterator iterTest = new MnistDataSetIterator(batchSizePerWorker, false, 12345);
            List<DataSet> trainDataList = new ArrayList<>();
            List<DataSet> testDataList = new ArrayList<>();
            int count = 0;
            while (iterTrain.hasNext() && count++ < 30) {
                trainDataList.add(iterTrain.next());
            }
            while (iterTest.hasNext()) {
                testDataList.add(iterTest.next());
            }

            JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);
            JavaRDD<DataSet> testData = sc.parallelize(testDataList);


            //----------------------------------
            //Create network configuration and conduct network training
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .activation(Activation.LEAKYRELU)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Nesterovs(0.02, 0.9))
                    .l2(1e-4)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(28 * 28).nOut(500).build())
                    .layer(1, new DenseLayer.Builder().nIn(500).nOut(100).build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation(Activation.SOFTMAX).nIn(100).nOut(10).build())
                    .pretrain(false).backprop(true)
                    .build();

            //Configuration for Spark training: see https://deeplearning4j.org/distributed for explanation of these configuration options

            TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)
                    .averagingFrequency(2)
                    .build();

            //Create the Spark network
            SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);
            sparkNet.setDefaultEvaluationWorkers(evalWorkers);

            //Execute training:
            for (int i = 0; i < numEpochs; i++) {
                sparkNet.fit(trainData);
            }

            //Perform evaluation (distributed)
            Evaluation evaluation = sparkNet.evaluate(testData);
            log.info("***** Evaluation *****");
            log.info(evaluation.stats());

            //Delete the temp training files, now that we are done with them
            tm.deleteTempFiles(sc);

            assertEquals(10000, evaluation.getNumRowCounter()); //10k test set
            assertTrue(!Double.isNaN(evaluation.accuracy()));
            assertTrue(evaluation.accuracy() >= 0.10);
            assertTrue(evaluation.precision() >= 0.10);
            assertTrue(evaluation.recall() >= 0.10);
            assertTrue(evaluation.f1() >= 0.10);
        }
    }


}
