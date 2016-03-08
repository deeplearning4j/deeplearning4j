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



import org.apache.spark.api.java.JavaPairRDD;
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
import scala.Tuple2;

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
                .list()
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
                .list()
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
                .list()
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
                .list()
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
        assertMapEquals(evalExpected.falseNegatives(),evalActual.falseNegatives());
        assertMapEquals(evalExpected.falsePositives(), evalActual.falsePositives());
        assertMapEquals(evalExpected.trueNegatives(), evalActual.trueNegatives());
        assertMapEquals(evalExpected.truePositives(),evalActual.truePositives());
        assertEquals(evalExpected.precision(), evalActual.precision(), 1e-3);
        assertEquals(evalExpected.recall(), evalActual.recall(), 1e-3);
        assertEquals(evalExpected.getConfusionMatrix(), evalActual.getConfusionMatrix());
    }

    private static void assertMapEquals(Map<Integer,Integer> first, Map<Integer,Integer> second){
        assertEquals(first.keySet(),second.keySet());
        for( Integer i : first.keySet()){
            assertEquals(first.get(i),second.get(i));
        }
    }

    @Test
    public void testSmallAmountOfData(){
        //Idea: Test spark training where some executors don't get any data
        //in this case: by having fewer examples (2 DataSets) than executors (local[*])

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(Updater.RMSPROP)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .list()
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

    @Test
    public void testDistributedScoring(){

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .regularization(true).l1(0.1).l2(0.1)
                .seed(123)
                .updater(Updater.NESTEROVS)
                .learningRate(0.1)
                .momentum(0.9)
                .list()
                .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                        .nIn(nIn).nOut(3)
                        .activation("tanh").build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(3).nOut(nOut)
                        .activation("softmax")
                        .build())
                .backprop(true)
                .pretrain(false)
                .build();

        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc,conf);
        MultiLayerNetwork netCopy = sparkNet.getNetwork().clone();

        int nRows = 100;

        INDArray features = Nd4j.rand(nRows,nIn);
        INDArray labels = Nd4j.zeros(nRows, nOut);
        Random r = new Random(12345);
        for( int i=0; i<nRows; i++ ){
            labels.putScalar(new int[]{i,r.nextInt(nOut)},1.0);
        }

        INDArray localScoresWithReg = netCopy.scoreExamples(new DataSet(features,labels),true);
        INDArray localScoresNoReg = netCopy.scoreExamples(new DataSet(features,labels),false);

        List<Tuple2<String,DataSet>> dataWithKeys = new ArrayList<>();
        for( int i=0; i<nRows; i++ ){
            DataSet ds = new DataSet(features.getRow(i).dup(),labels.getRow(i).dup());
            dataWithKeys.add(new Tuple2<>(String.valueOf(i),ds));
        }
        JavaPairRDD<String,DataSet> dataWithKeysRdd = sc.parallelizePairs(dataWithKeys);

        JavaPairRDD<String,Double> sparkScoresWithReg = sparkNet.scoreExamples(dataWithKeysRdd, true, 4);
        JavaPairRDD<String,Double> sparkScoresNoReg = sparkNet.scoreExamples(dataWithKeysRdd,false,4);

        Map<String,Double> sparkScoresWithRegMap = sparkScoresWithReg.collectAsMap();
        Map<String,Double> sparkScoresNoRegMap = sparkScoresNoReg.collectAsMap();

        for( int i=0; i<nRows; i++ ){
            double scoreRegExp = localScoresWithReg.getDouble(i);
            double scoreRegAct = sparkScoresWithRegMap.get(String.valueOf(i));
            assertEquals(scoreRegExp,scoreRegAct,1e-5);

            double scoreNoRegExp = localScoresNoReg.getDouble(i);
            double scoreNoRegAct = sparkScoresNoRegMap.get(String.valueOf(i));
            assertEquals(scoreNoRegExp, scoreNoRegAct, 1e-5);

//            System.out.println(scoreRegExp + "\t" + scoreRegAct + "\t" + scoreNoRegExp + "\t" + scoreNoRegAct);
        }

        List<DataSet> dataNoKeys = new ArrayList<>();
        for( int i=0; i<nRows; i++ ){
            dataNoKeys.add(new DataSet(features.getRow(i).dup(),labels.getRow(i).dup()));
        }
        JavaRDD<DataSet> dataNoKeysRdd = sc.parallelize(dataNoKeys);

        List<Double> scoresWithReg = sparkNet.scoreExamples(dataNoKeysRdd,true,4).collect();
        List<Double> scoresNoReg = sparkNet.scoreExamples(dataNoKeysRdd,false,4).collect();
        Collections.sort(scoresWithReg);
        Collections.sort(scoresNoReg);
        double[] localScoresWithRegDouble = localScoresWithReg.data().asDouble();
        double[] localScoresNoRegDouble = localScoresNoReg.data().asDouble();
        Arrays.sort(localScoresWithRegDouble);
        Arrays.sort(localScoresNoRegDouble);

        for( int i=0; i<localScoresWithRegDouble.length; i++ ){
            assertEquals(localScoresWithRegDouble[i],scoresWithReg.get(i),1e-5);
            assertEquals(localScoresNoRegDouble[i],scoresNoReg.get(i),1e-5);

            //System.out.println(localScoresWithRegDouble[i] + "\t" + scoresWithReg.get(i) + "\t" + localScoresNoRegDouble[i] + "\t" + scoresNoReg.get(i));
        }
    }
}
