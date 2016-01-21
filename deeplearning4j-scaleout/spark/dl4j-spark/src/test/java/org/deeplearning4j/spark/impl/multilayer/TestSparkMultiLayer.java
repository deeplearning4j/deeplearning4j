/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.spark.impl.multilayer;



import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileInputStream;
import java.util.*;

import static org.junit.Assert.assertEquals;


/**
 * Created by agibsonccc on 1/18/15.
 */
public class TestSparkMultiLayer extends BaseSparkTest {


    @Test
    public void testFromSvmLightBackprop() throws Exception {
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), new ClassPathResource("svmLight/iris_svmLight_0.txt").getFile().getAbsolutePath()).toJavaRDD().map(new Function<LabeledPoint, LabeledPoint>() {
            @Override
            public LabeledPoint call(LabeledPoint v1) throws Exception {
                return new LabeledPoint(v1.label(), Vectors.dense(v1.features().toArray()));
            }
        }).cache();
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        DataSet d = new IrisDataSetIterator(150,150).next();
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(10)
                .list(2)
                .layer(0, new DenseLayer.Builder()
                        .nIn(4).nOut(100)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(100).nOut(3)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .backprop(true)
                .build();



        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        System.out.println("Initializing network");
        SparkDl4jMultiLayer master = new SparkDl4jMultiLayer(sc,conf);

        MultiLayerNetwork network2 = master.fit(data, 10);
        Evaluation evaluation = new Evaluation();
        evaluation.eval(d.getLabels(), network2.output(d.getFeatureMatrix()));
        System.out.println(evaluation.stats());


    }


    @Test
    public void testFromSvmLight() throws Exception {
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), new ClassPathResource("svmLight/iris_svmLight_0.txt").getFile().getAbsolutePath()).toJavaRDD().map(new Function<LabeledPoint, LabeledPoint>() {
            @Override
            public LabeledPoint call(LabeledPoint v1) throws Exception {
                return new LabeledPoint(v1.label(), Vectors.dense(v1.features().toArray()));
            }
        }).cache();

        DataSet d = new IrisDataSetIterator(150,150).next();
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(100).miniBatch(true)
                .maxNumLineSearchIterations(10)
                .list(2)
                .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(4).nOut(100)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(100).nOut(3)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .backprop(false)
                .build();



        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        System.out.println("Initializing network");
        SparkDl4jMultiLayer master = new SparkDl4jMultiLayer(sc,conf);

        MultiLayerNetwork network2 = master.fit(data, 10);
        Evaluation evaluation = new Evaluation();
        evaluation.eval(d.getLabels(), network2.output(d.getFeatureMatrix()));
        System.out.println(evaluation.stats());


    }


    @Test
    public void testIris2() throws Exception {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .momentum(0.9).seed(123)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(100)
                .maxNumLineSearchIterations(10)
                .list(2)
                .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(4).nOut(3)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(3).nOut(3)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .backprop(false)
                .build();



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
        Nd4j.writeTxt(params, writeTo.getAbsolutePath(), ",");
        INDArray load = Nd4j.read(new FileInputStream(writeTo.getAbsolutePath()));
        assertEquals(params,load);
        writeTo.delete();
        Evaluation evaluation = new Evaluation();
        evaluation.eval(d.getLabels(), network2.output(d.getFeatureMatrix()));
        System.out.println(evaluation.stats());
    }

    @Test
    public void testStaticInvocation() throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list(2)
                .layer(0, new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                        .nIn(4).nOut(3)
                        .activation("tanh").build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(3).nOut(3)
                        .activation("softmax")
                        .build())
                .build();

        DataSet dataSet = new IrisDataSetIterator(5,5).next();
        List<DataSet> list = dataSet.asList();
        JavaRDD<DataSet> data = sc.parallelize(list);
        JavaRDD<LabeledPoint> mllLibData = MLLibUtil.fromDataSet(sc, data);

        MultiLayerNetwork network = SparkDl4jMultiLayer.train(mllLibData, conf);
        INDArray params = network.params(true);
        File writeTo = new File(UUID.randomUUID().toString());
        Nd4j.writeTxt(params,writeTo.getAbsolutePath(),",");
        INDArray load = Nd4j.readTxt(writeTo.getAbsolutePath());
        assertEquals(params,load);
        writeTo.delete();

        String json = network.getLayerWiseConfigurations().toJson();
        MultiLayerConfiguration conf2 = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf, conf2);

        MultiLayerNetwork network3 = new MultiLayerNetwork(conf2);
        network3.init();
        network3.setParameters(params);
        INDArray params4 = network3.params(true);
        assertEquals(params, params4);
    }

    @Test
    public void testRunIteration() {

        DataSet dataSet = new IrisDataSetIterator(5,5).next();
        List<DataSet> list = dataSet.asList();
        JavaRDD<DataSet> data = sc.parallelize(list);

        SparkDl4jMultiLayer sparkNetCopy = new SparkDl4jMultiLayer(sc, getBasicConf());
        MultiLayerNetwork networkCopy = sparkNetCopy.fitDataSet(data);

        INDArray expectedParams = networkCopy.params();

        SparkDl4jMultiLayer sparkNet = getBasicNetwork();
        MultiLayerNetwork network = sparkNet.fitDataSet(data);
        INDArray actualParams = network.params();

        assertEquals(expectedParams.size(1), actualParams.size(1));
    }

    @Test
    public void testUpdaters() {
        SparkDl4jMultiLayer sparkNet = getBasicNetwork();
        MultiLayerNetwork netCopy = sparkNet.getNetwork().clone();

        netCopy.fit(data);
        Updater expectedUpdater = netCopy.conf().getLayer().getUpdater();
        double expectedLR = netCopy.conf().getLayer().getLearningRate();
        double expectedMomentum = netCopy.conf().getLayer().getMomentum();

        Updater actualUpdater = sparkNet.getNetwork().conf().getLayer().getUpdater();
        sparkNet.runIteration(sparkData);
        double actualLR = sparkNet.getNetwork().conf().getLayer().getLearningRate();
        double actualMomentum = sparkNet.getNetwork().conf().getLayer().getMomentum();

        assertEquals(expectedUpdater, actualUpdater);
        assertEquals(expectedLR, actualLR, 0.01);
        assertEquals(expectedMomentum, actualMomentum, 0.01);

    }


    @Test
    public void testEvaluation(){

        SparkDl4jMultiLayer sparkNet = getBasicNetwork();
        MultiLayerNetwork netCopy = sparkNet.getNetwork().clone();

        Evaluation evalExpected = new Evaluation();
        INDArray outLocal = netCopy.output(input, Layer.TrainingMode.TEST);
        evalExpected.eval(labels, outLocal);

        Evaluation evalActual = sparkNet.evaluate(sparkData);

        assertEquals(evalExpected.accuracy(), evalActual.accuracy(), 1e-3);
        assertEquals(evalExpected.f1(), evalActual.f1(), 1e-3);
        assertEquals(evalExpected.getNumRowCounter(), evalActual.getNumRowCounter(), 1e-3);
        assertEquals(evalExpected.falseNegatives(),evalActual.falseNegatives(),1e-3);
        assertEquals(evalExpected.falsePositives(), evalActual.falsePositives(), 1e-3);
        assertEquals(evalExpected.trueNegatives(),evalActual.trueNegatives(),1e-3);
        assertEquals(evalExpected.truePositives(),evalActual.truePositives(),1e-3);
        assertEquals(evalExpected.precision(), evalActual.precision(), 1e-3);
        assertEquals(evalExpected.recall(), evalActual.recall(), 1e-3);
        assertEquals(evalExpected.getConfusionMatrix(), evalActual.getConfusionMatrix());
    }

    @Test
    public void testSmallAmountOfData(){
        //Idea: Test spark training where some executors don't get any data
        //in this case: by having fewer examples (2 DataSets) than executors (local[*])

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(Updater.RMSPROP)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .list(2)
                .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                        .nIn(nIn).nOut(3)
                        .activation("tanh").build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(3).nOut(nOut)
                        .activation("softmax")
                        .build())
                .build();

        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc,conf);

        Nd4j.getRandom().setSeed(12345);
        DataSet d1 = new DataSet(Nd4j.rand(1,nIn),Nd4j.rand(1,nOut));
        DataSet d2 = new DataSet(Nd4j.rand(1,nIn),Nd4j.rand(1,nOut));

        JavaRDD<DataSet> rddData = sc.parallelize(Arrays.asList(d1,d2));

        sparkNet.fitDataSet(rddData);

    }

}
