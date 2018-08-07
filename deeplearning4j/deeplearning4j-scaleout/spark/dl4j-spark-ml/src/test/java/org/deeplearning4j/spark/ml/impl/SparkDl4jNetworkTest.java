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

package org.deeplearning4j.spark.ml.impl;


import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.SQLContext;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.ml.utils.DatasetFacade;
import org.deeplearning4j.spark.ml.utils.ParamSerializer;
import org.junit.After;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.UUID;

public class SparkDl4jNetworkTest {

    private SparkConf sparkConf = new SparkConf().setAppName("testing").setMaster("local[4]");
    private JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
    private SQLContext sqlContext = new SQLContext(sparkContext);

    @Test
    public void testNetwork() {
        DatasetFacade df = DatasetFacade.dataRows(sqlContext.read().json("src/test/resources/dl4jnetwork"));
        Pipeline p = new Pipeline().setStages(new PipelineStage[] {getAssembler(new String[] {"x", "y"}, "features")});
        DatasetFacade part2 = DatasetFacade.dataRows(p.fit(df.get()).transform(df.get()).select("features", "label"));

        ParamSerializer ps = new ParamHelper();
        MultiLayerConfiguration mc = getNNConfiguration();
        Collection<TrainingListener> il = new ArrayList<>();
        il.add(new ScoreIterationListener(1));

        SparkDl4jNetwork sparkDl4jNetwork =
                        new SparkDl4jNetwork(mc, 2, ps, 1, il, true).setFeaturesCol("features").setLabelCol("label");

        SparkDl4jModel sm = sparkDl4jNetwork.fit(part2.get());
        MultiLayerNetwork mln = sm.getMultiLayerNetwork();
        Assert.assertNotNull(mln);
        DatasetFacade transformed = DatasetFacade.dataRows(sm.transform(part2.get()));
        List<?> rows = transformed.get().collectAsList();
        Assert.assertNotNull(sm.getTrainingStats());
        Assert.assertNotNull(rows);
    }

    @Test
    public void testNetworkLoader() throws Exception {
        DatasetFacade df = DatasetFacade.dataRows(sqlContext.read().json("src/test/resources/dl4jnetwork"));
        Pipeline p = new Pipeline().setStages(new PipelineStage[] {getAssembler(new String[] {"x", "y"}, "features")});
        DatasetFacade part2 = DatasetFacade.dataRows(p.fit(df.get()).transform(df.get()).select("features", "label"));

        ParamSerializer ps = new ParamHelper();
        MultiLayerConfiguration mc = getNNConfiguration();
        Collection<TrainingListener> il = new ArrayList<>();
        il.add(new ScoreIterationListener(1));

        SparkDl4jNetwork sparkDl4jNetwork =
                        new SparkDl4jNetwork(mc, 2, ps, 1, il, true).setFeaturesCol("features").setLabelCol("label");

        String fileName = UUID.randomUUID().toString();
        SparkDl4jModel sm = sparkDl4jNetwork.fit(part2.get());
        sm.write().overwrite().save(fileName);
        SparkDl4jModel spdm = SparkDl4jModel.load(fileName);
        Assert.assertNotNull(spdm);

        File file1 = new File(fileName);
        File file2 = new File(fileName + "_metadata");
        FileUtils.deleteDirectory(file1);
        FileUtils.deleteDirectory(file2);
    }

    @After
    public void closeIt() {
        sparkContext.close();
    }

    private MultiLayerConfiguration getNNConfiguration() {
        return new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.UNIFORM).updater(new Nesterovs(0.1)).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(100).weightInit(WeightInit.XAVIER)
                                        .activation(Activation.RELU).build())
                        .layer(1, new DenseLayer.Builder().nIn(100).nOut(120).weightInit(WeightInit.XAVIER)
                                        .activation(Activation.RELU).build())
                        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SOFTMAX).nIn(120)
                                        .nOut(2).build())
                        .pretrain(false).backprop(true).build();
    }

    private static VectorAssembler getAssembler(String[] input, String output) {
        return new VectorAssembler().setInputCols(input).setOutputCol(output);
    }

    static public class ParamHelper implements ParamSerializer {

        public ParameterAveragingTrainingMaster apply() {
            return new ParameterAveragingTrainingMaster.Builder(2).averagingFrequency(2).workerPrefetchNumBatches(2)
                            .batchSizePerWorker(2).build();
        }
    }
}
