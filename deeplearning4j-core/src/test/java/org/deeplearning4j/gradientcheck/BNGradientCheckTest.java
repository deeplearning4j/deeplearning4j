package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 *
 */
public class BNGradientCheckTest {
    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-5;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-5;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-9;

    static {
        //Force Nd4j initialization, then set data type to double:
        Nd4j.zeros(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testGradient2dSimple(){
        DataSet ds = new IrisDataSetIterator(150,150).next();
        ds.normalizeZeroMeanZeroUnitVariance();
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .learningRate(1.0)
                .regularization(false)
                .updater(Updater.NONE)
                .seed(12345L)
                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4).nOut(3)
                        .activation("identity")
                        .build())
                .layer(1, new BatchNormalization.Builder()
                        .nOut(3)
                        .build())
                .layer(2, new ActivationLayer.Builder().activation("tanh").build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax")
                        .nIn(3).nOut(3)
                        .build())
                .pretrain(false).backprop(true);

        MultiLayerNetwork mln = new MultiLayerNetwork(builder.build());
        mln.init();

        if (PRINT_RESULTS) {
            for (int j = 0; j < mln.getnLayers(); j++)
                System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

        assertTrue(gradOK);
    }

    @Test
    public void testGradientCnnSimple(){
        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int depth = 1;
        int hw = 4;
        int nOut = 4;
        INDArray input = Nd4j.rand(new int[]{minibatch, depth, hw, hw});
        INDArray labels = Nd4j.zeros(minibatch, nOut);
        Random r = new Random(12345);
        for( int i=0; i<minibatch; i++ ){
            labels.putScalar(i,r.nextInt(nOut),1.0);
        }

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .learningRate(1.0)
                .regularization(false)
                .updater(Updater.NONE)
                .seed(12345L)
                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 2))
                .list()
                .layer(0, new ConvolutionLayer.Builder()
                        .kernelSize(2,2)
                        .stride(1,1)
                        .nIn(depth).nOut(2)
                        .activation("identity")
                        .build())
                .layer(1, new BatchNormalization.Builder().build())
                .layer(2, new ActivationLayer.Builder().activation("tanh").build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax")
                        .nOut(nOut)
                        .build())
                .setInputType(InputType.convolutional(hw,hw,depth))
                .pretrain(false).backprop(true);

        MultiLayerNetwork mln = new MultiLayerNetwork(builder.build());
        mln.init();

        if (PRINT_RESULTS) {
            for (int j = 0; j < mln.getnLayers(); j++)
                System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

        assertTrue(gradOK);
    }

    @Test
    public void testGradientBNWithCNNandSubsampling(){
        //Parameterized test, testing combinations of:
        // (a) activation function
        // (b) Whether to test at random initialization, or after some learning (i.e., 'characteristic mode of operation')
        // (c) Loss function (with specified output activations)
        // (d) l1 and l2 values
        String[] activFns = {"sigmoid","tanh","identity"};
        boolean[] characteristic = {false,true};	//If true: run some backprop steps first

        LossFunctions.LossFunction[] lossFunctions = {LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD, LossFunctions.LossFunction.MSE};
        String[] outputActivations = {"softmax", "tanh"};	//i.e., lossFunctions[i] used with outputActivations[i] here

        double[] l2vals = {0.0, 0.1, 0.1};
        double[] l1vals = {0.0, 0.0, 0.2};	//i.e., use l2vals[j] with l1vals[j]

        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int depth = 2;
        int hw = 5;
        int nOut = 3;
        INDArray input = Nd4j.rand(new int[]{minibatch, depth, hw, hw});
        INDArray labels = Nd4j.zeros(minibatch, nOut);
        Random r = new Random(12345);
        for( int i=0; i<minibatch; i++ ){
            labels.putScalar(i,r.nextInt(nOut),1.0);
        }

        DataSet ds = new DataSet(input,labels);

        for( String afn : activFns) {
            for(boolean doLearningFirst : characteristic) {
                for(int i = 0; i < lossFunctions.length; i++) {
                    for (int j = 0; j < l2vals.length; j++) {
                        LossFunctions.LossFunction lf = lossFunctions[i];
                        String outputActivation = outputActivations[i];

                        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                                .seed(12345)
                                .regularization(l1vals[j] > 0 || l2vals[j] > 0)
                                .l1(l1vals[j]).l2(l2vals[j])
                                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                                .updater(Updater.NONE)
                                .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(-2, 2))
                                .seed(12345L)
                                .list()
                                .layer(0, new ConvolutionLayer.Builder(2, 2)
                                        .stride(1, 1)
                                        .nOut(3)
                                        .activation(afn)
                                        .build())
                                .layer(1, new BatchNormalization.Builder().build())
                                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                        .kernelSize(2, 2).stride(1, 1).build())
                                .layer(3, new BatchNormalization())
                                .layer(4, new ActivationLayer.Builder().activation(afn).build())
                                .layer(5, new OutputLayer.Builder(lf)
                                        .activation(outputActivation)
                                        .nOut(nOut)
                                        .build())
                                .setInputType(InputType.convolutional(hw, hw, depth))
                                .pretrain(false).backprop(true);

                        MultiLayerConfiguration conf = builder.build();

                        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                        mln.init();
                        String name = new Object() {}.getClass().getEnclosingMethod().getName();

                        if (doLearningFirst) {
                            //Run a number of iterations of learning
                            mln.setInput(ds.getFeatures());
                            mln.setLabels(ds.getLabels());
                            mln.computeGradientAndScore();
                            double scoreBefore = mln.score();
                            for (int k = 0; k < 5; k++)
                                mln.fit(ds);
                            mln.computeGradientAndScore();
                            double scoreAfter = mln.score();
                            //Can't test in 'characteristic mode of operation' if not learning
                            String msg = name + " - score did not (sufficiently) decrease during learning - activationFn="
                                    + afn + ", lossFn=" + lf + ", outputActivation=" + outputActivation + ", doLearningFirst= " + doLearningFirst
                                    + " (before=" + scoreBefore + ", scoreAfter=" + scoreAfter + ")";
                            assertTrue(msg, scoreAfter < 0.9 * scoreBefore);
                        }

                        if (PRINT_RESULTS) {
                            System.out.println(name + " - activationFn=" + afn + ", lossFn=" + lf + ", outputActivation=" + outputActivation
                                    + ", doLearningFirst=" + doLearningFirst + ", l1=" + l1vals[j] + ", l2=" + l2vals[j]);
                            for (int k = 0; k < mln.getnLayers(); k++)
                                System.out.println("Layer " + k + " # params: " + mln.getLayer(k).numParams());
                        }

                        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                        assertTrue(gradOK);
                    }
                }
            }
        }
    }


    @Test
    public void testGradientDense(){
        //Parameterized test, testing combinations of:
        // (a) activation function
        // (b) Whether to test at random initialization, or after some learning (i.e., 'characteristic mode of operation')
        // (c) Loss function (with specified output activations)
        // (d) l1 and l2 values
        String[] activFns = {"sigmoid","tanh","identity"};
        boolean[] characteristic = {false,true};	//If true: run some backprop steps first

        LossFunctions.LossFunction[] lossFunctions = {LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD, LossFunctions.LossFunction.MSE};
        String[] outputActivations = {"softmax", "tanh"};	//i.e., lossFunctions[i] used with outputActivations[i] here

        double[] l2vals = {0.0, 0.1, 0.1};
        double[] l1vals = {0.0, 0.0, 0.2};	//i.e., use l2vals[j] with l1vals[j]

        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int nIn = 5;
        int nOut = 3;
        INDArray input = Nd4j.rand(new int[]{minibatch, nIn});
        INDArray labels = Nd4j.zeros(minibatch, nOut);
        Random r = new Random(12345);
        for( int i=0; i<minibatch; i++ ){
            labels.putScalar(i,r.nextInt(nOut),1.0);
        }

        DataSet ds = new DataSet(input,labels);

        for( String afn : activFns) {
            for(boolean doLearningFirst : characteristic) {
                for(int i = 0; i < lossFunctions.length; i++) {
                    for (int j = 0; j < l2vals.length; j++) {
                        LossFunctions.LossFunction lf = lossFunctions[i];
                        String outputActivation = outputActivations[i];

                        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                                .regularization(l1vals[j] > 0 || l2vals[j] > 0)
                                .l1(l1vals[j]).l2(l2vals[j])
                                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                                .updater(Updater.NONE)
                                .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(-2, 2))
                                .seed(12345L)
                                .list()
                                .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(4).activation(afn).build())
                                .layer(1, new BatchNormalization.Builder().build())
                                .layer(2, new DenseLayer.Builder().nIn(4).nOut(4).build())
                                .layer(3, new BatchNormalization())
                                .layer(4, new OutputLayer.Builder(lf).activation(outputActivation).nOut(nOut).build())
                                .pretrain(false).backprop(true);

                        MultiLayerConfiguration conf = builder.build();

                        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                        mln.init();

                        String name = new Object() {}.getClass().getEnclosingMethod().getName();

                        if (doLearningFirst) {
                            //Run a number of iterations of learning
                            mln.setInput(ds.getFeatures());
                            mln.setLabels(ds.getLabels());
                            mln.computeGradientAndScore();
                            double scoreBefore = mln.score();
                            for (int k = 0; k < 10; k++)
                                mln.fit(ds);
                            mln.computeGradientAndScore();
                            double scoreAfter = mln.score();
                            //Can't test in 'characteristic mode of operation' if not learning
                            String msg = name + " - score did not (sufficiently) decrease during learning - activationFn="
                                    + afn + ", lossFn=" + lf + ", outputActivation=" + outputActivation + ", doLearningFirst= " + doLearningFirst
                                    + " (before=" + scoreBefore + ", scoreAfter=" + scoreAfter + ")";
                            assertTrue(msg, scoreAfter < 0.8 * scoreBefore);
                        }

                        if (PRINT_RESULTS) {
                            System.out.println(name + " - activationFn=" + afn + ", lossFn=" + lf + ", outputActivation=" + outputActivation
                                    + ", doLearningFirst=" + doLearningFirst + ", l1=" + l1vals[j] + ", l2=" + l2vals[j]);
                            for (int k = 0; k < mln.getnLayers(); k++)
                                System.out.println("Layer " + k + " # params: " + mln.getLayer(k).numParams());
                        }

                        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                        assertTrue(gradOK);
                    }
                }
            }
        }
    }

    @Test
    public void testGradient2dFixedGammaBeta(){
        DataSet ds = new IrisDataSetIterator(150,150).next();
        ds.normalizeZeroMeanZeroUnitVariance();
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .learningRate(1.0)
                .regularization(false)
                .updater(Updater.NONE)
                .seed(12345L)
                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4).nOut(3)
                        .activation("identity")
                        .build())
                .layer(1, new BatchNormalization.Builder()
                        .lockGammaBeta(true).gamma(2.0).beta(0.5)
                        .nOut(3)
                        .build())
                .layer(2, new ActivationLayer.Builder().activation("tanh").build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax")
                        .nIn(3).nOut(3)
                        .build())
                .pretrain(false).backprop(true);

        MultiLayerNetwork mln = new MultiLayerNetwork(builder.build());
        mln.init();

        if (PRINT_RESULTS) {
            for (int j = 0; j < mln.getnLayers(); j++)
                System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

        assertTrue(gradOK);
    }

    @Test
    public void testGradientCnnFixedGammaBeta(){
        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int depth = 1;
        int hw = 4;
        int nOut = 4;
        INDArray input = Nd4j.rand(new int[]{minibatch, depth, hw, hw});
        INDArray labels = Nd4j.zeros(minibatch, nOut);
        Random r = new Random(12345);
        for( int i=0; i<minibatch; i++ ){
            labels.putScalar(i,r.nextInt(nOut),1.0);
        }

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .learningRate(1.0)
                .regularization(false)
                .updater(Updater.NONE)
                .seed(12345L)
                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 2))
                .list()
                .layer(0, new ConvolutionLayer.Builder()
                        .kernelSize(2,2)
                        .stride(1,1)
                        .nIn(depth).nOut(2)
                        .activation("identity")
                        .build())
                .layer(1, new BatchNormalization.Builder().lockGammaBeta(true).gamma(2.0).beta(0.5).build())
                .layer(2, new ActivationLayer.Builder().activation("tanh").build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax")
                        .nOut(nOut)
                        .build())
                .setInputType(InputType.convolutional(hw,hw,depth))
                .pretrain(false).backprop(true);

        MultiLayerNetwork mln = new MultiLayerNetwork(builder.build());
        mln.init();

        if (PRINT_RESULTS) {
            for (int j = 0; j < mln.getnLayers(); j++)
                System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

        assertTrue(gradOK);
    }
}
