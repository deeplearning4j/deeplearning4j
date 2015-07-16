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

package org.deeplearning4j.nn.multilayer;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Lists;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.iterationlistener.LossPlotterIterationListener;
import org.deeplearning4j.plot.iterationlistener.NeuralNetPlotterIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.*;

/**
 * Created by agibsonccc on 12/27/14.
 */
public class MultiLayerTest {

    private static final Logger log = LoggerFactory.getLogger(MultiLayerTest.class);

    @Test
    public void testSetParams() {
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .nIn(4)
                .nOut(3)
                .layer(new org.deeplearning4j.nn.conf.layers.RBM())
                .activationFunction("tanh")
                .list(2)
                .hiddenLayerSizes(3)
                .override(1, new ClassifierOverride(1))
                .build();

        MultiLayerNetwork network3 = new MultiLayerNetwork(conf);
        network3.init();

        INDArray params = network3.params();
        network3.setParameters(params);
        network3.setScore();
        INDArray params4 = network3.params();
        assertEquals(params,params4);
    }

    @Test
    public void testDbnFaces() {
        DataSetIterator iter = new LFWDataSetIterator(28,28);

        DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .nIn(next.numInputs()).nOut(next.numOutcomes())
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .constrainGradientToUnitNorm(true)
                .numLineSearchIterations(5)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0,1e-5))
                .iterations(5).learningRate(1e-3)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN).hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .layer(new RBM())
                .list(4).hiddenLayerSizes(600,250,100).override(3,new ClassifierOverride()).build();
        
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(Arrays.<IterationListener>asList(new ScoreIterationListener(4),new NeuralNetPlotterIterationListener(4)));
        network.fit(next);

    }




    @Test
    public void testBackProp() {
        Nd4j.getRandom().setSeed(123);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(5).weightInit(WeightInit.XAVIER)
                .seed(123)
                .activationFunction("tanh")
                .nIn(4).nOut(3)
                .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer())
                .list(3).backward(true).pretrain(false)
                .hiddenLayerSizes(new int[]{3,2}).override(2, new ConfOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.activationFunction("softmax");
                        builder.layer(new org.deeplearning4j.nn.conf.layers.OutputLayer());
                        builder.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
                    }
                }).build();


        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(Lists.<IterationListener>newArrayList(new ScoreIterationListener(1)));

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();
        SplitTestAndTrain trainTest = next.splitTestAndTrain(110);
        network.setInput(trainTest.getTrain().getFeatureMatrix());
        network.setLabels(trainTest.getTrain().getLabels());
        network.init();
        network.fit(trainTest.getTrain());

        DataSet test = trainTest.getTest();
        Evaluation eval = new Evaluation();
        INDArray output = network.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        log.info("Score " + eval.stats());

    }

    @Test
    public void testDbn() throws Exception {
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(100)
                .layer(new org.deeplearning4j.nn.conf.layers.RBM())
                .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0,1))
                .activationFunction("tanh").momentum(0.9)
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .constrainGradientToUnitNorm(true).k(1).regularization(true).l2(2e-4)
                .visibleUnit(org.deeplearning4j.nn.conf.layers.RBM.VisibleUnit.GAUSSIAN).hiddenUnit(org.deeplearning4j.nn.conf.layers.RBM.HiddenUnit.RECTIFIED)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .nIn(4).nOut(3).list(2)
                .hiddenLayerSizes(3)
                .override(1, new ClassifierOverride(1)).build();

            NeuralNetConfiguration conf2 = new NeuralNetConfiguration.Builder()
                    .layer(new org.deeplearning4j.nn.conf.layers.RBM())
                    .nIn(784).nOut(600).applySparsity(true).sparsity(0.1)
                    .build();

        Layer l = LayerFactories.getFactory(conf2).create(conf2,
                Arrays.<IterationListener>asList(new ScoreIterationListener(2)),0);

        MultiLayerNetwork d = new MultiLayerNetwork(conf);

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        DataSet next = iter.next();

        Nd4j.writeTxt(next.getFeatureMatrix(),"iris.txt","\t");

        next.normalizeZeroMeanZeroUnitVariance();

        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(110);
        DataSet train = testAndTrain.getTrain();

        d.fit(train);

        DataSet test = testAndTrain.getTest();

        Evaluation eval = new Evaluation();
        INDArray output = d.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        log.info("Score " + eval.stats());

    }
    
    @Test
    public void testMultiLayerNinNout(){
    	//[Nin,Nout]: [4,13], [13,3]
    	int[] hiddenLayerSizes1 = {13};
    	int[] expNin1 =  {4,13};
    	int[] expNout1 = {13,3};
    	MultiLayerConfiguration conf1 = getNinNoutIrisConfig(hiddenLayerSizes1);
    	MultiLayerNetwork net1 = new MultiLayerNetwork(conf1);
    	net1.init();
    	checkNinNoutForEachLayer( expNin1, expNout1, conf1, net1 );
    	
    	int[] hiddenLayerSizes2 = {5,7};
    	int[] expNin2 =  {4,5,7};
    	int[] expNout2 = {5,7,3};
    	MultiLayerConfiguration conf2 = getNinNoutIrisConfig(hiddenLayerSizes2);
    	MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
    	net2.init();
    	checkNinNoutForEachLayer( expNin2, expNout2, conf2, net2 );
    	
    	int[] hiddenLayerSizes3 = {5,7,9};
    	int[] expNin3 =  {4,5,7,9};
    	int[] expNout3 = {5,7,9,3};
    	MultiLayerConfiguration conf3 = getNinNoutIrisConfig(hiddenLayerSizes3);
    	MultiLayerNetwork net3 = new MultiLayerNetwork(conf3);
    	net3.init();
    	checkNinNoutForEachLayer( expNin3, expNout3, conf3, net3 );
    	
    	int[] hiddenLayerSizes4 = {5,7,9,11,13,15,17};
    	int[] expNin4 =  {4,5,7,9,11,13,15,17};
    	int[] expNout4 = {5,7,9,11,13,15,17,3};
    	MultiLayerConfiguration conf4 = getNinNoutIrisConfig(hiddenLayerSizes4);
    	MultiLayerNetwork net4 = new MultiLayerNetwork(conf4);
    	net4.init();
    	checkNinNoutForEachLayer( expNin4, expNout4, conf4, net4 );
    }
    
    private static MultiLayerConfiguration getNinNoutIrisConfig( int[] hiddenLayerSizes ){
    	MultiLayerConfiguration c = new NeuralNetConfiguration.Builder()
		.nIn(4).nOut(3)
		.layer(new RBM())
		.list(hiddenLayerSizes.length+1).hiddenLayerSizes(hiddenLayerSizes)
		.override(hiddenLayerSizes.length, new ClassifierOverride())
		.build();
		return c;
    }
    
    private static void checkNinNoutForEachLayer( int[] expNin, int[] expNout, MultiLayerConfiguration conf,
    		MultiLayerNetwork network ){
    	
    	//Check configuration
    	for( int i=0; i<expNin.length; i++ ){
    		NeuralNetConfiguration layerConf = conf.getConf(i);
    		assertTrue(layerConf.getNIn() == expNin[i]);
    		assertTrue(layerConf.getNOut() == expNout[i]);
    	}
    	
    	//Check Layer
    	for( int i=0; i<expNin.length; i++ ){
    		Layer layer = network.getLayers()[i];
    		assertTrue(layer.conf().getNIn() == expNin[i]);
    		assertTrue(layer.conf().getNOut() == expNout[i]);
    		int[] weightShape = layer.getParam(DefaultParamInitializer.WEIGHT_KEY).shape();
    		assertTrue(weightShape[0]==expNin[i]);
    		assertTrue(weightShape[1]==expNout[i]);
    	}
    }
    


    @Test
    public void testGradientWithAsList(){
        MultiLayerNetwork net1 = new MultiLayerNetwork(getConf());
        MultiLayerNetwork net2 = new MultiLayerNetwork(getConf());
        net1.init();
        net2.init();

        DataSet x1 = new IrisDataSetIterator(1,150).next();
        DataSet all = new IrisDataSetIterator(150,150).next();
        DataSet x2 = all.asList().get(0);

        //x1 and x2 contain identical data
        assertArrayEquals(asFloat(x1.getFeatureMatrix()), asFloat(x2.getFeatureMatrix()), 0.0f);  //OK
        assertArrayEquals(asFloat(x1.getLabels()), asFloat(x2.getLabels()), 0.0f);                //OK
        assertEquals(x1,x2);    //Fails, DataSet doesn't override Object.equals()

        //Set inputs/outputs so gradient can be calculated:
        net1.feedForward(x1.getFeatureMatrix());
        net2.feedForward(x2.getFeatureMatrix());
        ((OutputLayer)net1.getLayers()[1]).setLabels(x1.getLabels());
        ((OutputLayer)net2.getLayers()[1]).setLabels(x2.getLabels());

        net1.gradient();        //OK
        net2.gradient();        //IllegalArgumentException: Buffers must fill up specified length 29
    }


    @Test
    public void testFeedForwardActivationsAndDerivatives(){
        MultiLayerNetwork network = new MultiLayerNetwork(getConf());
        network.init();
        DataSet data = new IrisDataSetIterator(1,150).next();
        network.fit(data);
        Pair result = network.feedForwardActivationsAndDerivatives();
        List<INDArray> first = (List) result.getFirst();
        List<INDArray> second = (List) result.getSecond();
        assertEquals(first.size(), second.size());
    }


    private static MultiLayerConfiguration getConf(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(4).nOut(3)
                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1))
                .seed(12345L)
                .list(2)
                .hiddenLayerSizes(5)
                .override(1, new ClassifierOverride())
                .build();
        return conf;
    }

    public static float[] asFloat( INDArray arr ){
        int len = arr.length();
        float[] f = new float[len];
        for( int i=0; i<len; i++ ) f[i] = arr.getFloat(i);
        return f;
    }


}
