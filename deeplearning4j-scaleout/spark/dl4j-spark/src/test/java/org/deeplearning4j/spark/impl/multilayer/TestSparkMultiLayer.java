/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.spark.impl.multilayer;



import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.canova.api.records.reader.impl.SVMLightRecordReader;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.stepfunctions.GradientStepFunction;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.junit.Test;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.List;
import java.util.UUID;

import static org.junit.Assert.assertEquals;


/**
 * Created by agibsonccc on 1/18/15.
 */
public class TestSparkMultiLayer extends BaseSparkTest {

    private static final Logger log = LoggerFactory.getLogger(TestSparkMultiLayer.class);


    @Test
    public void testIris() throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .nIn(4).nOut(3).layerFactory(LayerFactories.getFactory(RBM.class)).visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .activationFunction(Activations.tanh()).list(2).hiddenLayerSizes(3)
                .override(new ConfOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        if (i == 1) {
                            builder.activationFunction(Activations.softMaxRows());
                            builder.layerFactory(LayerFactories.getFactory(OutputLayer.class));
                            builder.lossFunction(LossFunctions.LossFunction.MCXENT);
                        }
                    }
                }).build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        int numParams = network.numParams();
        INDArray params = network.params();
        assertEquals(numParams,params.length());
        SparkDl4jMultiLayer sparkDl4jMultiLayer = new SparkDl4jMultiLayer(sc,conf);
        String path = new ClassPathResource("data/svmLight/iris_svmLight_0.txt").getFile().toURI().toString();
        sparkDl4jMultiLayer.fit(path,4,new SVMLightRecordReader());

    }

    @Test
    public void testIris2() throws Exception {


        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
               .momentum(0.9).constrainGradientToUnitNorm(true).iterationListener(new ScoreIterationListener(10))
                .activationFunction(Activations.tanh()).stepFunction(new GradientStepFunction())
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).dropOut(0.3)
                .iterations(100).visibleUnit(RBM.VisibleUnit.GAUSSIAN).batchSize(10)
                .l2(2e-4).regularization(true).weightInit(WeightInit.VI)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .nIn(4).nOut(3).layerFactory(LayerFactories.getFactory(RBM.class))
                .list(3).hiddenLayerSizes(3,2)
                .override(new ConfOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {

                        if (i == 2) {
                            builder.activationFunction(Activations.softMaxRows());
                            builder.layerFactory(LayerFactories.getFactory(OutputLayer.class));
                            builder.lossFunction(LossFunctions.LossFunction.MCXENT);
                        }
                    }
                }).build();


        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        System.out.println("Initializing network");
        SparkDl4jMultiLayer master = new SparkDl4jMultiLayer(sc,conf);
        DataSet d = new IrisDataSetIterator(150,150).next();
        d.normalizeZeroMeanZeroUnitVariance();
        d.shuffle();
        List<DataSet> next = d.asList();


        JavaRDD<DataSet> data = sc.parallelize(next);



        MultiLayerNetwork network2 = master.fitDataSet(data);

        INDArray params = network2.params();
        File writeTo = new File(UUID.randomUUID().toString());
        Nd4j.writeTxt(params,writeTo.getAbsolutePath(),",");
        INDArray load = Nd4j.readTxt(writeTo.getAbsolutePath(),",");
        assertEquals(params,load);
        writeTo.delete();
        Evaluation evaluation = new Evaluation();
        evaluation.eval(d.getLabels(), network2.output(d.getFeatureMatrix()));

    }

    @Test
    public void testStaticInvocation() throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .nIn(4).nOut(3).layerFactory(LayerFactories.getFactory(RBM.class))
                .activationFunction(Activations.tanh()).list(2).hiddenLayerSizes(3)
                .override(new ConfOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        if (i == 1) {
                            builder.activationFunction(Activations.softMaxRows());
                            builder.layerFactory(LayerFactories.getFactory(OutputLayer.class));
                            builder.lossFunction(LossFunctions.LossFunction.MCXENT);
                        }
                    }
                }).build();

        DataSet dataSet = new IrisDataSetIterator(150,150).next();
        List<DataSet> list = dataSet.asList();
        JavaRDD<DataSet> data = sc.parallelize(list);
        JavaRDD<LabeledPoint> mllLibData = MLLibUtil.fromDataSet(sc,data);

        MultiLayerNetwork network = SparkDl4jMultiLayer.train(mllLibData,conf);
        INDArray params = network.params();
        File writeTo = new File(UUID.randomUUID().toString());
        Nd4j.writeTxt(params,writeTo.getAbsolutePath(),",");
        INDArray load = Nd4j.readTxt(writeTo.getAbsolutePath(),",");
        assertEquals(params,load);
        writeTo.delete();

        String json = network.getLayerWiseConfigurations().toJson();
        MultiLayerConfiguration conf2 = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf,conf2);

        MultiLayerNetwork network3 = new MultiLayerNetwork(conf2);
        network3.init();
        network3.setParameters(params);
        INDArray params4 = network3.params();
        assertEquals(params,params4);


    }

}
