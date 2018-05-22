package org.deeplearning4j.nn.layers.recurrent;

import junit.framework.TestCase;
import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.GravesBidirectionalLSTMParamInitializer;
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import static org.junit.Assert.*;


public class GravesBidirectionalLSTMTest extends BaseDL4JTest {
    private double score = 0.0;

    @Test
    public void testBidirectionalLSTMGravesForwardBasic() {
        //Very basic test of forward prop. of LSTM layer with a time series.
        //Essentially make sure it doesn't throw any exceptions, and provides output in the correct shape.
        int nIn = 13;
        int nHiddenUnits = 17;

        final NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                        .layer(new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder().nIn(nIn)
                                        .nOut(nHiddenUnits).activation(Activation.TANH).build())
                        .build();

        val numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        final GravesBidirectionalLSTM layer =
                        (GravesBidirectionalLSTM) conf.getLayer().instantiate(conf, null, 0, params, true);

        //Data: has shape [miniBatchSize,nIn,timeSeriesLength];
        //Output/activations has shape [miniBatchsize,nHiddenUnits,timeSeriesLength];

        final INDArray dataSingleExampleTimeLength1 = Nd4j.ones(1, nIn, 1);
        final INDArray activations1 = layer.activate(dataSingleExampleTimeLength1, false, LayerWorkspaceMgr.noWorkspaces());
        assertArrayEquals(activations1.shape(), new long[] {1, nHiddenUnits, 1});

        final INDArray dataMultiExampleLength1 = Nd4j.ones(10, nIn, 1);
        final INDArray activations2 = layer.activate(dataMultiExampleLength1, false, LayerWorkspaceMgr.noWorkspaces());
        assertArrayEquals(activations2.shape(), new long[] {10, nHiddenUnits, 1});

        final INDArray dataSingleExampleLength12 = Nd4j.ones(1, nIn, 12);
        final INDArray activations3 = layer.activate(dataSingleExampleLength12, false, LayerWorkspaceMgr.noWorkspaces());
        assertArrayEquals(activations3.shape(), new long[] {1, nHiddenUnits, 12});

        final INDArray dataMultiExampleLength15 = Nd4j.ones(10, nIn, 15);
        final INDArray activations4 = layer.activate(dataMultiExampleLength15, false, LayerWorkspaceMgr.noWorkspaces());
        assertArrayEquals(activations4.shape(), new long[] {10, nHiddenUnits, 15});
    }

    @Test
    public void testBidirectionalLSTMGravesBackwardBasic() {
        //Very basic test of backprop for mini-batch + time series
        //Essentially make sure it doesn't throw any exceptions, and provides output in the correct shape.

        testGravesBackwardBasicHelper(13, 3, 17, 10, 7);
        testGravesBackwardBasicHelper(13, 3, 17, 1, 7); //Edge case: miniBatchSize = 1
        testGravesBackwardBasicHelper(13, 3, 17, 10, 1); //Edge case: timeSeriesLength = 1
        testGravesBackwardBasicHelper(13, 3, 17, 1, 1); //Edge case: both miniBatchSize = 1 and timeSeriesLength = 1
    }

    private static void testGravesBackwardBasicHelper(int nIn, int nOut, int lstmNHiddenUnits, int miniBatchSize,
                    int timeSeriesLength) {

        INDArray inputData = Nd4j.ones(miniBatchSize, nIn, timeSeriesLength);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                        .layer(new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder().nIn(nIn)
                                        .nOut(lstmNHiddenUnits).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new UniformDistribution(0, 1)).activation(Activation.TANH).build())
                        .build();

        long numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        GravesBidirectionalLSTM lstm =
                        (GravesBidirectionalLSTM) conf.getLayer().instantiate(conf, null, 0, params, true);
        lstm.setBackpropGradientsViewArray(Nd4j.create(1, conf.getLayer().initializer().numParams(conf)));
        //Set input, do a forward pass:
        lstm.activate(inputData, false, LayerWorkspaceMgr.noWorkspaces());
        assertNotNull(lstm.input());

        INDArray epsilon = Nd4j.ones(miniBatchSize, lstmNHiddenUnits, timeSeriesLength);

        Pair<Gradient, INDArray> out = lstm.backpropGradient(epsilon, LayerWorkspaceMgr.noWorkspaces());
        Gradient outGradient = out.getFirst();
        INDArray nextEpsilon = out.getSecond();

        INDArray biasGradientF = outGradient.getGradientFor(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS);
        INDArray inWeightGradientF =
                        outGradient.getGradientFor(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS);
        INDArray recurrentWeightGradientF = outGradient
                        .getGradientFor(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS);
        assertNotNull(biasGradientF);
        assertNotNull(inWeightGradientF);
        assertNotNull(recurrentWeightGradientF);

        INDArray biasGradientB = outGradient.getGradientFor(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS);
        INDArray inWeightGradientB =
                        outGradient.getGradientFor(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS);
        INDArray recurrentWeightGradientB = outGradient
                        .getGradientFor(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS);
        assertNotNull(biasGradientB);
        assertNotNull(inWeightGradientB);
        assertNotNull(recurrentWeightGradientB);

        assertArrayEquals(biasGradientF.shape(), new long[] {1, 4 * lstmNHiddenUnits});
        assertArrayEquals(inWeightGradientF.shape(), new long[] {nIn, 4 * lstmNHiddenUnits});
        assertArrayEquals(recurrentWeightGradientF.shape(), new long[] {lstmNHiddenUnits, 4 * lstmNHiddenUnits + 3});

        assertArrayEquals(biasGradientB.shape(), new long[] {1, 4 * lstmNHiddenUnits});
        assertArrayEquals(inWeightGradientB.shape(), new long[] {nIn, 4 * lstmNHiddenUnits});
        assertArrayEquals(recurrentWeightGradientB.shape(), new long[] {lstmNHiddenUnits, 4 * lstmNHiddenUnits + 3});

        assertNotNull(nextEpsilon);
        assertArrayEquals(nextEpsilon.shape(), new long[] {miniBatchSize, nIn, timeSeriesLength});

        //Check update:
        for (String s : outGradient.gradientForVariable().keySet()) {
            lstm.update(outGradient.getGradientFor(s), s);
        }
    }

    @Test
    public void testGravesBidirectionalLSTMForwardPassHelper() throws Exception {
        //GravesBidirectionalLSTM.activateHelper() has different behaviour (due to optimizations) when forBackprop==true vs false
        //But should otherwise provide identical activations
        Nd4j.getRandom().setSeed(12345);

        final int nIn = 10;
        final int layerSize = 15;
        final int miniBatchSize = 4;
        final int timeSeriesLength = 7;

        final NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                        .layer(new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder().nIn(nIn)
                                        .nOut(layerSize).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new UniformDistribution(0, 1)).activation(Activation.TANH).build())
                        .build();

        long numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        final GravesBidirectionalLSTM lstm =
                        (GravesBidirectionalLSTM) conf.getLayer().instantiate(conf, null, 0, params, true);
        final INDArray input = Nd4j.rand(new int[] {miniBatchSize, nIn, timeSeriesLength});
        lstm.setInput(input, LayerWorkspaceMgr.noWorkspaces());


        final INDArray fwdPassFalse = LSTMHelpers.activateHelper(lstm, lstm.conf(), new ActivationSigmoid(),
                        lstm.input(),
                        lstm.getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS),
                        lstm.getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS),
                        lstm.getParam(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS), false, null, null,
                        false, true, GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS, null, true,
                        null, CacheMode.NONE, LayerWorkspaceMgr.noWorkspaces()).fwdPassOutput;

        final INDArray[] fwdPassTrue = LSTMHelpers.activateHelper(lstm, lstm.conf(), new ActivationSigmoid(),
                        lstm.input(),
                        lstm.getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS),
                        lstm.getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS),
                        lstm.getParam(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS), false, null, null,
                        true, true, GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS, null, true, null,
                        CacheMode.NONE, LayerWorkspaceMgr.noWorkspaces()).fwdPassOutputAsArrays;

        //I have no idea what the heck this does --Ben
        for (int i = 0; i < timeSeriesLength; i++) {
            final INDArray sliceFalse = fwdPassFalse.tensorAlongDimension(i, 1, 0);
            final INDArray sliceTrue = fwdPassTrue[i];
            assertTrue(sliceFalse.equals(sliceTrue));
        }
    }

    static private void reverseColumnsInPlace(final INDArray x) {
        final long N = x.size(1);
        final INDArray x2 = x.dup();

        for (int t = 0; t < N; t++) {
            final long b = N - t - 1;
            //clone?
            x.putColumn(t, x2.getColumn(b));
        }
    }

    @Test
    public void testGetSetParmas() {
        final int nIn = 2;
        final int layerSize = 3;
        final int miniBatchSize = 2;
        final int timeSeriesLength = 10;

        Nd4j.getRandom().setSeed(12345);

        final NeuralNetConfiguration confBidirectional = new NeuralNetConfiguration.Builder()
                        .layer(new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder().nIn(nIn)
                                        .nOut(layerSize).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new UniformDistribution(-0.1, 0.1)).activation(Activation.TANH).build())
                        .build();


        long numParams = confBidirectional.getLayer().initializer().numParams(confBidirectional);
        INDArray params = Nd4j.create(1, numParams);
        final GravesBidirectionalLSTM bidirectionalLSTM = (GravesBidirectionalLSTM) confBidirectional.getLayer()
                        .instantiate(confBidirectional, null, 0, params, true);


        final INDArray sig = Nd4j.rand(new int[] {miniBatchSize, nIn, timeSeriesLength});

        final INDArray act1 = bidirectionalLSTM.activate(sig, false, LayerWorkspaceMgr.noWorkspaces());

        params = bidirectionalLSTM.params();

        bidirectionalLSTM.setParams(params);

        final INDArray act2 = bidirectionalLSTM.activate(sig, false, LayerWorkspaceMgr.noWorkspaces());

        assertArrayEquals(act2.data().asDouble(), act1.data().asDouble(), 1e-8);


    }

    @Test
    public void testSimpleForwardsAndBackwardsActivation() {

        final int nIn = 2;
        final int layerSize = 3;
        final int miniBatchSize = 1;
        final int timeSeriesLength = 5;

        Nd4j.getRandom().setSeed(12345);

        final NeuralNetConfiguration confBidirectional =
                        new NeuralNetConfiguration.Builder()
                                        .layer(new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder()
                                                        .nIn(nIn).nOut(layerSize).weightInit(WeightInit.DISTRIBUTION)
                                                        .dist(new UniformDistribution(-0.1, 0.1))
                                                        .activation(Activation.TANH).updater(new NoOp()).build())
                                        .build();

        final NeuralNetConfiguration confForwards = new NeuralNetConfiguration.Builder()
                        .layer(new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(nIn).nOut(layerSize)
                                        .weightInit(WeightInit.ZERO).activation(Activation.TANH).build())
                        .build();

        long numParams = confForwards.getLayer().initializer().numParams(confForwards);
        INDArray params = Nd4j.create(1, numParams);
        long numParamsBD = confBidirectional.getLayer().initializer().numParams(confBidirectional);
        INDArray paramsBD = Nd4j.create(1, numParamsBD);
        final GravesBidirectionalLSTM bidirectionalLSTM = (GravesBidirectionalLSTM) confBidirectional.getLayer()
                        .instantiate(confBidirectional, null, 0, paramsBD, true);
        final GravesLSTM forwardsLSTM =
                        (GravesLSTM) confForwards.getLayer().instantiate(confForwards, null, 0, params, true);

        bidirectionalLSTM.setBackpropGradientsViewArray(
                        Nd4j.create(1, confBidirectional.getLayer().initializer().numParams(confBidirectional)));
        forwardsLSTM.setBackpropGradientsViewArray(
                        Nd4j.create(1, confForwards.getLayer().initializer().numParams(confForwards)));


        final INDArray sig = Nd4j.rand(new int[] {miniBatchSize, nIn, timeSeriesLength});
        final INDArray sigb = sig.dup();
        reverseColumnsInPlace(sigb.slice(0));

        final INDArray recurrentWeightsF = bidirectionalLSTM
                        .getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS);
        final INDArray inputWeightsF =
                        bidirectionalLSTM.getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS);
        final INDArray biasWeightsF =
                        bidirectionalLSTM.getParam(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS);

        final INDArray recurrentWeightsF2 = forwardsLSTM.getParam(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY);
        final INDArray inputWeightsF2 = forwardsLSTM.getParam(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY);
        final INDArray biasWeightsF2 = forwardsLSTM.getParam(GravesLSTMParamInitializer.BIAS_KEY);

        //assert that the forwards part of the bidirectional layer is equal to that of the regular LSTM
        assertArrayEquals(recurrentWeightsF2.shape(), recurrentWeightsF.shape());
        assertArrayEquals(inputWeightsF2.shape(), inputWeightsF.shape());
        assertArrayEquals(biasWeightsF2.shape(), biasWeightsF.shape());

        forwardsLSTM.setParam(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY, recurrentWeightsF);
        forwardsLSTM.setParam(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY, inputWeightsF);
        forwardsLSTM.setParam(GravesLSTMParamInitializer.BIAS_KEY, biasWeightsF);

        //copy forwards weights to make the forwards activations do the same thing

        final INDArray recurrentWeightsB = bidirectionalLSTM
                        .getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS);
        final INDArray inputWeightsB =
                        bidirectionalLSTM.getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS);
        final INDArray biasWeightsB =
                        bidirectionalLSTM.getParam(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS);

        //assert that the forwards and backwards are the same shapes
        assertArrayEquals(recurrentWeightsF.shape(), recurrentWeightsB.shape());
        assertArrayEquals(inputWeightsF.shape(), inputWeightsB.shape());
        assertArrayEquals(biasWeightsF.shape(), biasWeightsB.shape());

        //zero out backwards layer
        bidirectionalLSTM.setParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS,
                        Nd4j.zeros(recurrentWeightsB.shape()));
        bidirectionalLSTM.setParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS,
                        Nd4j.zeros(inputWeightsB.shape()));
        bidirectionalLSTM.setParam(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS,
                        Nd4j.zeros(biasWeightsB.shape()));


        forwardsLSTM.setInput(sig, LayerWorkspaceMgr.noWorkspaces());

        //compare activations
        final INDArray activation1 = forwardsLSTM.activate(sig, false, LayerWorkspaceMgr.noWorkspaces()).slice(0);
        final INDArray activation2 = bidirectionalLSTM.activate(sig, false, LayerWorkspaceMgr.noWorkspaces()).slice(0);

        assertArrayEquals(activation1.data().asFloat(), activation2.data().asFloat(), 1e-5f);

        final INDArray randSig = Nd4j.rand(new int[] {1, layerSize, timeSeriesLength});
        final INDArray randSigBackwards = randSig.dup();
        reverseColumnsInPlace(randSigBackwards.slice(0));


        final Pair<Gradient, INDArray> backprop1 = forwardsLSTM.backpropGradient(randSig, LayerWorkspaceMgr.noWorkspaces());
        final Pair<Gradient, INDArray> backprop2 = bidirectionalLSTM.backpropGradient(randSig, LayerWorkspaceMgr.noWorkspaces());

        //compare gradients
        assertArrayEquals(
                        backprop1.getFirst().getGradientFor(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY).dup()
                                        .data().asFloat(),
                        backprop2.getFirst()
                                        .getGradientFor(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS)
                                        .dup().data().asFloat(),
                        1e-5f);

        assertArrayEquals(
                        backprop1.getFirst().getGradientFor(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY).dup().data()
                                        .asFloat(),
                        backprop2.getFirst()
                                        .getGradientFor(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS)
                                        .dup().data().asFloat(),
                        1e-5f);

        assertArrayEquals(
                        backprop1.getFirst().getGradientFor(GravesLSTMParamInitializer.BIAS_KEY).dup().data().asFloat(),
                        backprop2.getFirst().getGradientFor(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS)
                                        .dup().data().asFloat(),
                        1e-5f);

        //copy forwards to backwards
        bidirectionalLSTM.setParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS,
                        bidirectionalLSTM.getParam(
                                        GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS));

        bidirectionalLSTM.setParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS,
                        bidirectionalLSTM.getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS));

        bidirectionalLSTM.setParam(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS,
                        bidirectionalLSTM.getParam(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS));

        //zero out forwards layer
        bidirectionalLSTM.setParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS,
                        Nd4j.zeros(recurrentWeightsB.shape()));
        bidirectionalLSTM.setParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS,
                        Nd4j.zeros(inputWeightsB.shape()));
        bidirectionalLSTM.setParam(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS,
                        Nd4j.zeros(biasWeightsB.shape()));

        //run on reversed signal
        final INDArray activation3 = bidirectionalLSTM.activate(sigb, false, LayerWorkspaceMgr.noWorkspaces()).slice(0);

        final INDArray activation3Reverse = activation3.dup();
        reverseColumnsInPlace(activation3Reverse);

        assertEquals(activation3Reverse, activation1);
        assertArrayEquals(activation3Reverse.shape(), activation1.shape());

        //test backprop now
        final INDArray refBackGradientReccurrent =
                        backprop1.getFirst().getGradientFor(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY);

        final INDArray refBackGradientInput =
                        backprop1.getFirst().getGradientFor(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY);

        final INDArray refBackGradientBias = backprop1.getFirst().getGradientFor(GravesLSTMParamInitializer.BIAS_KEY);

        //reverse weights only with backwards signal should yield same result as forwards weights with forwards signal
        final Pair<Gradient, INDArray> backprop3 = bidirectionalLSTM.backpropGradient(randSigBackwards, LayerWorkspaceMgr.noWorkspaces());

        final INDArray backGradientRecurrent = backprop3.getFirst()
                        .getGradientFor(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS);
        final INDArray backGradientInput = backprop3.getFirst()
                        .getGradientFor(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS);
        final INDArray backGradientBias =
                        backprop3.getFirst().getGradientFor(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS);

        assertArrayEquals(refBackGradientBias.dup().data().asDouble(), backGradientBias.dup().data().asDouble(), 1e-6);

        assertArrayEquals(refBackGradientInput.dup().data().asDouble(), backGradientInput.dup().data().asDouble(),
                        1e-6);

        assertArrayEquals(refBackGradientReccurrent.dup().data().asDouble(),
                        backGradientRecurrent.dup().data().asDouble(), 1e-6);

        final INDArray refEpsilon = backprop1.getSecond().dup();
        final INDArray backEpsilon = backprop3.getSecond().dup();

        reverseColumnsInPlace(refEpsilon.slice(0));
        assertArrayEquals(backEpsilon.dup().data().asDouble(), refEpsilon.dup().data().asDouble(), 1e-6);

    }

    @Test
    @Ignore
    public void testConvergence() {
        Nd4j.getRandom().setSeed(12345);
        final int state1Len = 100;
        final int state2Len = 30;

        //segment by signal mean
        //Data: has shape [miniBatchSize,nIn,timeSeriesLength];

        final INDArray sig1 = Nd4j.randn(new int[] {1, 2, state1Len}).mul(0.1);
        final INDArray sig2 = Nd4j.randn(new int[] {1, 2, state2Len}).mul(0.1)
                        .add(Nd4j.ones(new int[] {1, 2, state2Len}).mul(1.0));

        INDArray sig = Nd4j.concat(2, sig1, sig2);
        INDArray labels = Nd4j.zeros(new int[] {1, 2, state1Len + state2Len});

        for (int t = 0; t < state1Len; t++) {
            labels.putScalar(new int[] {0, 0, t}, 1.0);
        }

        for (int t = state1Len; t < state1Len + state2Len; t++) {
            labels.putScalar(new int[] {0, 1, t}, 1.0);
        }

        for (int i = 0; i < 3; i++) {
            sig = Nd4j.concat(2, sig, sig);
            labels = Nd4j.concat(2, labels, labels);
        }

        final DataSet ds = new DataSet(sig, labels);

        final MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new AdaGrad(0.1))
                        .l2(0.001)
                        .seed(12345).list().pretrain(false)
                        .layer(0, new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder()
                                        .activation(Activation.TANH).nIn(2).nOut(2).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new UniformDistribution(-0.05, 0.05)).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder()
                                        .activation(Activation.TANH).nIn(2).nOut(2).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new UniformDistribution(-0.05, 0.05)).build())
                        /*.layer(0, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().activation(Activation.TANH).nIn(2).nOut(2)
                                .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(-0.05,0.05))
                                .build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().activation(Activation.TANH).nIn(2).nOut(2)
                                .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(-0.05,0.05))
                                .build())*/ //this converges
                        .layer(2, new org.deeplearning4j.nn.conf.layers.RnnOutputLayer.Builder()
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(2).nOut(2)
                                        .activation(Activation.TANH).build())
                        .backprop(true).build();

        final MultiLayerNetwork net = new MultiLayerNetwork(conf);

        final TrainingListener scoreSaver = new BaseTrainingListener() {

            @Override
            public void iterationDone(Model model, int iteration, int epoch) {
                score = model.score();
            }
        };

        net.setListeners(scoreSaver, new ScoreIterationListener(1));
        double oldScore = Double.POSITIVE_INFINITY;
        net.init();
        for (int iEpoch = 0; iEpoch < 3; iEpoch++) {
            for( int i=0; i<5; i++ ) {
                net.fit(ds);
            }

            System.out.print(String.format("score is %f%n", score));

            assertTrue(!Double.isNaN(score));

            assertTrue(score < 0.9 * oldScore);
            oldScore = score;

            final INDArray output = net.output(ds.getFeatureMatrix());
            Evaluation evaluation = new Evaluation();
            evaluation.evalTimeSeries(ds.getLabels(), output);
            System.out.print(evaluation.stats() + "\n");
        }
    }

    @Test
    public void testSerialization() {

        final MultiLayerConfiguration conf1 = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new AdaGrad(0.1))
                        .l2(0.001)
                        .seed(12345).list().pretrain(false)
                        .layer(0, new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder()
                                        .activation(Activation.TANH).nIn(2).nOut(2).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new UniformDistribution(-0.05, 0.05)).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder()
                                        .activation(Activation.TANH).nIn(2).nOut(2).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new UniformDistribution(-0.05, 0.05)).build())
                        .layer(2, new org.deeplearning4j.nn.conf.layers.RnnOutputLayer.Builder()
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(2).nOut(2)
                                        .activation(Activation.TANH).build())
                        .backprop(true).build();


        final String json1 = conf1.toJson();

        final MultiLayerConfiguration conf2 = MultiLayerConfiguration.fromJson(json1);

        final String json2 = conf1.toJson();


        TestCase.assertEquals(json1, json2);
    }

    @Test
    public void testGateActivationFnsSanityCheck() {
        for (String gateAfn : new String[] {"sigmoid", "hardsigmoid"}) {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                            .seed(12345).list()
                            .layer(0, new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder()
                                            .gateActivationFunction(gateAfn).activation(Activation.TANH).nIn(2).nOut(2)
                                            .build())
                            .layer(1, new org.deeplearning4j.nn.conf.layers.RnnOutputLayer.Builder()
                                            .lossFunction(LossFunctions.LossFunction.MSE).nIn(2).nOut(2)
                                            .activation(Activation.TANH).build())
                            .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            assertEquals(gateAfn, ((org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM) net.getLayer(0).conf()
                            .getLayer()).getGateActivationFn().toString());

            INDArray in = Nd4j.rand(new int[] {3, 2, 5});
            INDArray labels = Nd4j.rand(new int[] {3, 2, 5});

            net.fit(in, labels);
        }
    }
}
