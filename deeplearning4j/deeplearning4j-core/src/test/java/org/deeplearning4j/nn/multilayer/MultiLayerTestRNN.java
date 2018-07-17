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

package org.deeplearning4j.nn.multilayer;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer;
import org.deeplearning4j.nn.layers.recurrent.GravesLSTM;
import org.deeplearning4j.nn.layers.recurrent.LSTM;
import org.deeplearning4j.nn.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

@Slf4j
public class MultiLayerTestRNN extends BaseDL4JTest {

    @Test
    public void testGravesLSTMInit() {
        int nIn = 8;
        int nOut = 25;
        int nHiddenUnits = 17;
        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder()
                                        .list().layer(0,
                                                        new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
                                                                        .nIn(nIn).nOut(nHiddenUnits)
                                                                        .weightInit(WeightInit.DISTRIBUTION)
                                                                        .activation(Activation.TANH).build())
                                        .layer(1, new RnnOutputLayer.Builder(LossFunction.MSE).nIn(nHiddenUnits)
                                                        .nOut(nOut).weightInit(WeightInit.DISTRIBUTION)
                                                        .activation(Activation.TANH).build())
                                        .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        //Ensure that we have the correct number weights and biases, that these have correct shape etc.
        Layer layer = network.getLayer(0);
        assertTrue(layer instanceof GravesLSTM);

        Map<String, INDArray> paramTable = layer.paramTable();
        assertTrue(paramTable.size() == 3); //2 sets of weights, 1 set of biases

        INDArray recurrentWeights = paramTable.get(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY);
        assertArrayEquals(recurrentWeights.shape(), new long[] {nHiddenUnits, 4 * nHiddenUnits + 3}); //Should be shape: [layerSize,4*layerSize+3]
        INDArray inputWeights = paramTable.get(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY);
        assertArrayEquals(inputWeights.shape(), new long[] {nIn, 4 * nHiddenUnits}); //Should be shape: [nIn,4*layerSize]
        INDArray biases = paramTable.get(GravesLSTMParamInitializer.BIAS_KEY);
        assertArrayEquals(biases.shape(), new long[] {1, 4 * nHiddenUnits}); //Should be shape: [1,4*layerSize]

        //Want forget gate biases to be initialized to > 0. See parameter initializer for details
        INDArray forgetGateBiases =
                        biases.get(NDArrayIndex.point(0), NDArrayIndex.interval(nHiddenUnits, 2 * nHiddenUnits));
        assertEquals(nHiddenUnits, (int) forgetGateBiases.gt(0).sum(Integer.MAX_VALUE).getDouble(0));

        val nParams = recurrentWeights.length() + inputWeights.length() + biases.length();
        assertTrue(nParams == layer.numParams());
    }

    @Test
    public void testGravesTLSTMInitStacked() {
        int nIn = 8;
        int nOut = 25;
        int[] nHiddenUnits = {17, 19, 23};
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(nIn).nOut(17)
                                        .weightInit(WeightInit.DISTRIBUTION).activation(Activation.TANH).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(17).nOut(19)
                                        .weightInit(WeightInit.DISTRIBUTION).activation(Activation.TANH).build())
                        .layer(2, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(19).nOut(23)
                                        .weightInit(WeightInit.DISTRIBUTION).activation(Activation.TANH).build())
                        .layer(3, new RnnOutputLayer.Builder(LossFunction.MSE).nIn(23).nOut(nOut)
                                        .weightInit(WeightInit.DISTRIBUTION).activation(Activation.TANH).build())
                        .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        //Ensure that we have the correct number weights and biases, that these have correct shape etc. for each layer
        for (int i = 0; i < nHiddenUnits.length; i++) {
            Layer layer = network.getLayer(i);
            assertTrue(layer instanceof GravesLSTM);

            Map<String, INDArray> paramTable = layer.paramTable();
            assertTrue(paramTable.size() == 3); //2 sets of weights, 1 set of biases

            int layerNIn = (i == 0 ? nIn : nHiddenUnits[i - 1]);

            INDArray recurrentWeights = paramTable.get(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY);
            assertArrayEquals(recurrentWeights.shape(), new long[] {nHiddenUnits[i], 4 * nHiddenUnits[i] + 3}); //Should be shape: [layerSize,4*layerSize+3]
            INDArray inputWeights = paramTable.get(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY);
            assertArrayEquals(inputWeights.shape(), new long[] {layerNIn, 4 * nHiddenUnits[i]}); //Should be shape: [nIn,4*layerSize]
            INDArray biases = paramTable.get(GravesLSTMParamInitializer.BIAS_KEY);
            assertArrayEquals(biases.shape(), new long[] {1, 4 * nHiddenUnits[i]}); //Should be shape: [1,4*layerSize]

            //Want forget gate biases to be initialized to > 0. See parameter initializer for details
            INDArray forgetGateBiases = biases.get(NDArrayIndex.point(0),
                            NDArrayIndex.interval(nHiddenUnits[i], 2 * nHiddenUnits[i]));
            assertEquals(nHiddenUnits[i], (int) forgetGateBiases.gt(0).sum(Integer.MAX_VALUE).getDouble(0));

            val nParams = recurrentWeights.length() + inputWeights.length() + biases.length();
            assertTrue(nParams == layer.numParams());
        }
    }

    @Test
    public void testRnnStateMethods() {
        Nd4j.getRandom().setSeed(12345);
        int timeSeriesLength = 6;

        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder()
                                        .list().layer(0,
                                                        new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
                                                                        .nIn(5).nOut(7).activation(Activation.TANH)
                                                                        .weightInit(WeightInit.DISTRIBUTION)
                                                                        .dist(new NormalDistribution(0, 0.5)).build())
                                        .layer(1, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(7)
                                                        .nOut(8).activation(Activation.TANH)
                                                        .weightInit(WeightInit.DISTRIBUTION)
                                                        .dist(new NormalDistribution(0,
                                                                        0.5))
                                                        .build())
                                        .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT)
                                                        .weightInit(WeightInit.DISTRIBUTION).nIn(8).nOut(4)
                                                        .activation(Activation.SOFTMAX)
                                                        .weightInit(WeightInit.DISTRIBUTION)
                                                        .dist(new NormalDistribution(0, 0.5)).build())
                                        .build();
        MultiLayerNetwork mln = new MultiLayerNetwork(conf);

        INDArray input = Nd4j.rand(new int[] {3, 5, timeSeriesLength});

        List<INDArray> allOutputActivations = mln.feedForward(input, true);
        INDArray outAct = allOutputActivations.get(3);

        INDArray outRnnTimeStep = mln.rnnTimeStep(input);

        assertTrue(outAct.equals(outRnnTimeStep)); //Should be identical here

        Map<String, INDArray> currStateL0 = mln.rnnGetPreviousState(0);
        Map<String, INDArray> currStateL1 = mln.rnnGetPreviousState(1);

        assertTrue(currStateL0.size() == 2);
        assertTrue(currStateL1.size() == 2);

        INDArray lastActL0 = currStateL0.get(GravesLSTM.STATE_KEY_PREV_ACTIVATION);
        INDArray lastMemL0 = currStateL0.get(GravesLSTM.STATE_KEY_PREV_MEMCELL);
        assertTrue(lastActL0 != null && lastMemL0 != null);

        INDArray lastActL1 = currStateL1.get(GravesLSTM.STATE_KEY_PREV_ACTIVATION);
        INDArray lastMemL1 = currStateL1.get(GravesLSTM.STATE_KEY_PREV_MEMCELL);
        assertTrue(lastActL1 != null && lastMemL1 != null);

        INDArray expectedLastActL0 = allOutputActivations.get(1).tensorAlongDimension(timeSeriesLength - 1, 1, 0);
        assertTrue(expectedLastActL0.equals(lastActL0));

        INDArray expectedLastActL1 = allOutputActivations.get(2).tensorAlongDimension(timeSeriesLength - 1, 1, 0);
        assertTrue(expectedLastActL1.equals(lastActL1));

        //Check clearing and setting of state:
        mln.rnnClearPreviousState();
        assertTrue(mln.rnnGetPreviousState(0).isEmpty());
        assertTrue(mln.rnnGetPreviousState(1).isEmpty());

        mln.rnnSetPreviousState(0, currStateL0);
        assertTrue(mln.rnnGetPreviousState(0).size() == 2);
        mln.rnnSetPreviousState(1, currStateL1);
        assertTrue(mln.rnnGetPreviousState(1).size() == 2);
    }

    @Test
    public void testRnnTimeStepLayers() {

        for( int layerType=0; layerType<3; layerType++ ) {
            org.deeplearning4j.nn.conf.layers.Layer l0;
            org.deeplearning4j.nn.conf.layers.Layer l1;
            String lastActKey;

            if(layerType == 0){
                l0 = new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(5).nOut(7)
                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                        .dist(new NormalDistribution(0, 0.5)).build();
                l1 = new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(7).nOut(8)
                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                        .dist(new NormalDistribution(0, 0.5)).build();
                lastActKey = GravesLSTM.STATE_KEY_PREV_ACTIVATION;
            } else if(layerType == 1){
                l0 = new org.deeplearning4j.nn.conf.layers.LSTM.Builder().nIn(5).nOut(7)
                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                        .dist(new NormalDistribution(0, 0.5)).build();
                l1 = new org.deeplearning4j.nn.conf.layers.LSTM.Builder().nIn(7).nOut(8)
                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                        .dist(new NormalDistribution(0, 0.5)).build();
                lastActKey = LSTM.STATE_KEY_PREV_ACTIVATION;
            } else {
                l0 = new org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn.Builder().nIn(5).nOut(7)
                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                        .dist(new NormalDistribution(0, 0.5)).build();
                l1 = new org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn.Builder().nIn(7).nOut(8)
                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                        .dist(new NormalDistribution(0, 0.5)).build();
                lastActKey = SimpleRnn.STATE_KEY_PREV_ACTIVATION;
            }

            log.info("Starting test for layer type: {}", l0.getClass().getSimpleName());


            Nd4j.getRandom().setSeed(12345);
            int timeSeriesLength = 12;

            //4 layer network: 2 GravesLSTM + DenseLayer + RnnOutputLayer. Hence also tests preprocessors.
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).list()
                    .layer(0, l0)
                    .layer(1, l1)
                    .layer(2, new DenseLayer.Builder().nIn(8).nOut(9).activation(Activation.TANH)
                            .weightInit(WeightInit.DISTRIBUTION).dist(
                                    new NormalDistribution(0,
                                            0.5))
                            .build())
                    .layer(3, new RnnOutputLayer.Builder(LossFunction.MCXENT).weightInit(WeightInit.DISTRIBUTION)
                            .nIn(9).nOut(4).activation(Activation.SOFTMAX)
                            .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.5))
                            .build())
                    .inputPreProcessor(2, new RnnToFeedForwardPreProcessor())
                    .inputPreProcessor(3, new FeedForwardToRnnPreProcessor()).build();
            MultiLayerNetwork mln = new MultiLayerNetwork(conf);

            INDArray input = Nd4j.rand(new int[]{3, 5, timeSeriesLength});

            List<INDArray> allOutputActivations = mln.feedForward(input, true);
            INDArray fullOutL0 = allOutputActivations.get(1);
            INDArray fullOutL1 = allOutputActivations.get(2);
            INDArray fullOutL3 = allOutputActivations.get(4);

            int[] inputLengths = {1, 2, 3, 4, 6, 12};

            //Do steps of length 1, then of length 2, ..., 12
            //Should get the same result regardless of step size; should be identical to standard forward pass
            for (int i = 0; i < inputLengths.length; i++) {
                int inLength = inputLengths[i];
                int nSteps = timeSeriesLength / inLength; //each of length inLength

                mln.rnnClearPreviousState();
                mln.setInputMiniBatchSize(1); //Reset; should be set by rnnTimeStep method

                for (int j = 0; j < nSteps; j++) {
                    int startTimeRange = j * inLength;
                    int endTimeRange = startTimeRange + inLength;

                    INDArray inputSubset;
                    if (inLength == 1) { //Workaround to nd4j bug
                        val sizes = new long[]{input.size(0), input.size(1), 1};
                        inputSubset = Nd4j.create(sizes);
                        inputSubset.tensorAlongDimension(0, 1, 0).assign(input.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.point(startTimeRange)));
                    } else {
                        inputSubset = input.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(startTimeRange, endTimeRange));
                    }
                    if (inLength > 1)
                        assertTrue(inputSubset.size(2) == inLength);

                    INDArray out = mln.rnnTimeStep(inputSubset);

                    INDArray expOutSubset;
                    if (inLength == 1) {
                        val sizes = new long[]{fullOutL3.size(0), fullOutL3.size(1), 1};
                        expOutSubset = Nd4j.create(sizes);
                        expOutSubset.tensorAlongDimension(0, 1, 0).assign(fullOutL3.get(NDArrayIndex.all(),
                                NDArrayIndex.all(), NDArrayIndex.point(startTimeRange)));
                    } else {
                        expOutSubset = fullOutL3.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(startTimeRange, endTimeRange));
                    }

                    assertEquals(expOutSubset, out);

                    Map<String, INDArray> currL0State = mln.rnnGetPreviousState(0);
                    Map<String, INDArray> currL1State = mln.rnnGetPreviousState(1);

                    INDArray lastActL0 = currL0State.get(lastActKey);
                    INDArray lastActL1 = currL1State.get(lastActKey);

                    INDArray expLastActL0 = fullOutL0.tensorAlongDimension(endTimeRange - 1, 1, 0);
                    INDArray expLastActL1 = fullOutL1.tensorAlongDimension(endTimeRange - 1, 1, 0);

                    assertEquals(expLastActL0, lastActL0);
                    assertEquals(expLastActL1, lastActL1);
                }
            }
        }
    }

    @Test
    public void testRnnTimeStep2dInput() {
        Nd4j.getRandom().setSeed(12345);
        int timeSeriesLength = 6;

        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder()
                                        .list().layer(0,
                                                        new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
                                                                        .nIn(5).nOut(7).activation(Activation.TANH)
                                                                        .weightInit(WeightInit.DISTRIBUTION)
                                                                        .dist(new NormalDistribution(0, 0.5)).build())
                                        .layer(1, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(7)
                                                        .nOut(8).activation(Activation.TANH)
                                                        .weightInit(WeightInit.DISTRIBUTION)
                                                        .dist(new NormalDistribution(0,
                                                                        0.5))
                                                        .build())
                                        .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT)
                                                        .weightInit(WeightInit.DISTRIBUTION).nIn(8).nOut(4)
                                                        .activation(Activation.SOFTMAX)
                                                        .weightInit(WeightInit.DISTRIBUTION)
                                                        .dist(new NormalDistribution(0, 0.5)).build())
                                        .build();
        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();

        INDArray input3d = Nd4j.rand(new long[] {3, 5, timeSeriesLength});
        INDArray out3d = mln.rnnTimeStep(input3d);
        assertArrayEquals(out3d.shape(), new long[] {3, 4, timeSeriesLength});

        mln.rnnClearPreviousState();
        for (int i = 0; i < timeSeriesLength; i++) {
            INDArray input2d = input3d.tensorAlongDimension(i, 1, 0);
            INDArray out2d = mln.rnnTimeStep(input2d);

            assertArrayEquals(out2d.shape(), new long[] {3, 4});

            INDArray expOut2d = out3d.tensorAlongDimension(i, 1, 0);
            assertEquals(out2d, expOut2d);
        }

        //Check same but for input of size [3,5,1]. Expect [3,4,1] out
        mln.rnnClearPreviousState();
        for (int i = 0; i < timeSeriesLength; i++) {
            INDArray temp = Nd4j.create(new int[] {3, 5, 1});
            temp.tensorAlongDimension(0, 1, 0).assign(input3d.tensorAlongDimension(i, 1, 0));
            INDArray out3dSlice = mln.rnnTimeStep(temp);
            assertArrayEquals(out3dSlice.shape(), new long[] {3, 4, 1});

            assertTrue(out3dSlice.tensorAlongDimension(0, 1, 0).equals(out3d.tensorAlongDimension(i, 1, 0)));
        }
    }

    @Test
    public void testTruncatedBPTTVsBPTT() {
        //Under some (limited) circumstances, we expect BPTT and truncated BPTT to be identical
        //Specifically TBPTT over entire data vector

        int timeSeriesLength = 12;
        int miniBatchSize = 7;
        int nIn = 5;
        int nOut = 4;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                .trainingWorkspaceMode(WorkspaceMode.NONE).inferenceWorkspaceMode(WorkspaceMode.NONE)
                .list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(nIn).nOut(7)
                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(7).nOut(8)
                                        .activation(Activation.TANH)
                                        .weightInit(WeightInit.DISTRIBUTION).dist(
                                                        new NormalDistribution(0,
                                                                        0.5))
                                        .build())
                        .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).weightInit(WeightInit.DISTRIBUTION)
                                        .nIn(8).nOut(nOut).activation(Activation.SOFTMAX)
                                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.5))
                                        .build())
                        .backprop(true).build();
        assertEquals(BackpropType.Standard, conf.getBackpropType());

        MultiLayerConfiguration confTBPTT = new NeuralNetConfiguration.Builder().seed(12345)
                .trainingWorkspaceMode(WorkspaceMode.NONE).inferenceWorkspaceMode(WorkspaceMode.NONE)
                .list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(nIn).nOut(7)
                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(7).nOut(8)
                                        .activation(Activation.TANH)
                                        .weightInit(WeightInit.DISTRIBUTION).dist(
                                                        new NormalDistribution(0,
                                                                        0.5))
                                        .build())
                        .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).weightInit(WeightInit.DISTRIBUTION)
                                        .nIn(8).nOut(nOut).activation(Activation.SOFTMAX)
                                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.5))
                                        .build())
                        .backprop(true).backpropType(BackpropType.TruncatedBPTT).tBPTTBackwardLength(timeSeriesLength)
                        .tBPTTForwardLength(timeSeriesLength).build();

        Nd4j.getRandom().setSeed(12345);
        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();

        Nd4j.getRandom().setSeed(12345);
        MultiLayerNetwork mlnTBPTT = new MultiLayerNetwork(confTBPTT);
        mlnTBPTT.init();

        mlnTBPTT.clearTbpttState = false;

        assertEquals(BackpropType.TruncatedBPTT, mlnTBPTT.getLayerWiseConfigurations().getBackpropType());
        assertEquals(timeSeriesLength, mlnTBPTT.getLayerWiseConfigurations().getTbpttFwdLength());
        assertEquals(timeSeriesLength, mlnTBPTT.getLayerWiseConfigurations().getTbpttBackLength());

        INDArray inputData = Nd4j.rand(new int[] {miniBatchSize, nIn, timeSeriesLength});
        INDArray labels = Nd4j.rand(new int[] {miniBatchSize, nOut, timeSeriesLength});

        mln.setInput(inputData);
        mln.setLabels(labels);

        mlnTBPTT.setInput(inputData);
        mlnTBPTT.setLabels(labels);

        mln.computeGradientAndScore();
        mlnTBPTT.computeGradientAndScore();

        Pair<Gradient, Double> mlnPair = mln.gradientAndScore();
        Pair<Gradient, Double> tbpttPair = mlnTBPTT.gradientAndScore();

        assertEquals(mlnPair.getFirst().gradientForVariable(), tbpttPair.getFirst().gradientForVariable());
        assertEquals(mlnPair.getSecond(), tbpttPair.getSecond());

        //Check states: expect stateMap to be empty but tBpttStateMap to not be
        Map<String, INDArray> l0StateMLN = mln.rnnGetPreviousState(0);
        Map<String, INDArray> l0StateTBPTT = mlnTBPTT.rnnGetPreviousState(0);
        Map<String, INDArray> l1StateMLN = mln.rnnGetPreviousState(0);
        Map<String, INDArray> l1StateTBPTT = mlnTBPTT.rnnGetPreviousState(0);

        Map<String, INDArray> l0TBPTTStateMLN = ((BaseRecurrentLayer<?>) mln.getLayer(0)).rnnGetTBPTTState();
        Map<String, INDArray> l0TBPTTStateTBPTT = ((BaseRecurrentLayer<?>) mlnTBPTT.getLayer(0)).rnnGetTBPTTState();
        Map<String, INDArray> l1TBPTTStateMLN = ((BaseRecurrentLayer<?>) mln.getLayer(1)).rnnGetTBPTTState();
        Map<String, INDArray> l1TBPTTStateTBPTT = ((BaseRecurrentLayer<?>) mlnTBPTT.getLayer(1)).rnnGetTBPTTState();

        assertTrue(l0StateMLN.isEmpty());
        assertTrue(l0StateTBPTT.isEmpty());
        assertTrue(l1StateMLN.isEmpty());
        assertTrue(l1StateTBPTT.isEmpty());

        assertTrue(l0TBPTTStateMLN.isEmpty());
        assertEquals(2, l0TBPTTStateTBPTT.size());
        assertTrue(l1TBPTTStateMLN.isEmpty());
        assertEquals(2, l1TBPTTStateTBPTT.size());

        INDArray tbpttActL0 = l0TBPTTStateTBPTT.get(GravesLSTM.STATE_KEY_PREV_ACTIVATION);
        INDArray tbpttActL1 = l1TBPTTStateTBPTT.get(GravesLSTM.STATE_KEY_PREV_ACTIVATION);

        List<INDArray> activations = mln.feedForward(inputData, true);
        INDArray l0Act = activations.get(1);
        INDArray l1Act = activations.get(2);
        INDArray expL0Act = l0Act.tensorAlongDimension(timeSeriesLength - 1, 1, 0);
        INDArray expL1Act = l1Act.tensorAlongDimension(timeSeriesLength - 1, 1, 0);
        assertEquals(tbpttActL0, expL0Act);
        assertEquals(tbpttActL1, expL1Act);
    }

    @Test
    public void testRnnActivateUsingStoredState() {
        int timeSeriesLength = 12;
        int miniBatchSize = 7;
        int nIn = 5;
        int nOut = 4;

        int nTimeSlices = 5;

        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder().seed(12345).list().layer(0,
                                        new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(nIn).nOut(7)
                                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                                        .dist(new NormalDistribution(0, 0.5)).build())
                                        .layer(1, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(7)
                                                        .nOut(8).activation(Activation.TANH)
                                                        .weightInit(WeightInit.DISTRIBUTION)
                                                        .dist(new NormalDistribution(0,
                                                                        0.5))
                                                        .build())
                                        .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT)
                                                        .weightInit(WeightInit.DISTRIBUTION).nIn(8).nOut(nOut)
                                                        .activation(Activation.SOFTMAX)
                                                        .weightInit(WeightInit.DISTRIBUTION)
                                                        .dist(new NormalDistribution(0, 0.5)).build())
                                        .build();

        Nd4j.getRandom().setSeed(12345);
        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();

        INDArray inputLong = Nd4j.rand(new int[] {miniBatchSize, nIn, nTimeSlices * timeSeriesLength});
        INDArray input = inputLong.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, timeSeriesLength));

        List<INDArray> outStandard = mln.feedForward(input, true);
        List<INDArray> outRnnAct = mln.rnnActivateUsingStoredState(input, true, true);

        //As initially state is zeros: expect these to be the same
        assertEquals(outStandard, outRnnAct);

        //Furthermore, expect multiple calls to this function to be the same:
        for (int i = 0; i < 3; i++) {
            assertEquals(outStandard, mln.rnnActivateUsingStoredState(input, true, true));
        }

        List<INDArray> outStandardLong = mln.feedForward(inputLong, true);
        BaseRecurrentLayer<?> l0 = ((BaseRecurrentLayer<?>) mln.getLayer(0));
        BaseRecurrentLayer<?> l1 = ((BaseRecurrentLayer<?>) mln.getLayer(1));

        for (int i = 0; i < nTimeSlices; i++) {
            INDArray inSlice = inputLong.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.interval(i * timeSeriesLength, (i + 1) * timeSeriesLength));
            List<INDArray> outSlice = mln.rnnActivateUsingStoredState(inSlice, true, true);
            List<INDArray> expOut = new ArrayList<>();
            for (INDArray temp : outStandardLong) {
                expOut.add(temp.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(i * timeSeriesLength, (i + 1) * timeSeriesLength)));
            }

            for (int j = 0; j < expOut.size(); j++) {
                INDArray exp = expOut.get(j);
                INDArray act = outSlice.get(j);
                System.out.println(j);
                System.out.println(exp.sub(act));
                assertEquals(exp, act);
            }

            assertEquals(expOut, outSlice);

            //Again, expect multiple calls to give the same output
            for (int j = 0; j < 3; j++) {
                outSlice = mln.rnnActivateUsingStoredState(inSlice, true, true);
                assertEquals(expOut, outSlice);
            }

            l0.rnnSetPreviousState(l0.rnnGetTBPTTState());
            l1.rnnSetPreviousState(l1.rnnGetTBPTTState());
        }
    }

    @Test
    public void testTruncatedBPTTSimple() {
        //Extremely simple test of the 'does it throw an exception' variety
        int timeSeriesLength = 12;
        int miniBatchSize = 7;
        int nIn = 5;
        int nOut = 4;

        int nTimeSlices = 20;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(nIn).nOut(7)
                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(7).nOut(8)
                                        .activation(Activation.TANH)
                                        .weightInit(WeightInit.DISTRIBUTION).dist(
                                                        new NormalDistribution(0,
                                                                        0.5))
                                        .build())
                        .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).weightInit(WeightInit.DISTRIBUTION)
                                        .nIn(8).nOut(nOut).activation(Activation.SOFTMAX)
                                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.5))
                                        .build())
                        .pretrain(false).backprop(true).backpropType(BackpropType.TruncatedBPTT)
                        .tBPTTBackwardLength(timeSeriesLength).tBPTTForwardLength(timeSeriesLength).build();

        Nd4j.getRandom().setSeed(12345);
        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();

        INDArray inputLong = Nd4j.rand(new int[] {miniBatchSize, nIn, nTimeSlices * timeSeriesLength});
        INDArray labelsLong = Nd4j.rand(new int[] {miniBatchSize, nOut, nTimeSlices * timeSeriesLength});

        mln.fit(inputLong, labelsLong);
    }

    @Test
    public void testTruncatedBPTTWithMasking() {
        //Extremely simple test of the 'does it throw an exception' variety
        int timeSeriesLength = 100;
        int tbpttLength = 10;
        int miniBatchSize = 7;
        int nIn = 5;
        int nOut = 4;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(nIn).nOut(7)
                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(7).nOut(8)
                                        .activation(Activation.TANH)
                                        .weightInit(WeightInit.DISTRIBUTION).dist(
                                                        new NormalDistribution(0,
                                                                        0.5))
                                        .build())
                        .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).weightInit(WeightInit.DISTRIBUTION)
                                        .nIn(8).nOut(nOut).activation(Activation.SOFTMAX)
                                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.5))
                                        .build())
                        .pretrain(false).backprop(true).backpropType(BackpropType.TruncatedBPTT)
                        .tBPTTBackwardLength(tbpttLength).tBPTTForwardLength(tbpttLength).build();

        Nd4j.getRandom().setSeed(12345);
        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();

        INDArray features = Nd4j.rand(new int[] {miniBatchSize, nIn, timeSeriesLength});
        INDArray labels = Nd4j.rand(new int[] {miniBatchSize, nOut, timeSeriesLength});

        INDArray maskArrayInput = Nd4j.ones(miniBatchSize, timeSeriesLength);
        INDArray maskArrayOutput = Nd4j.ones(miniBatchSize, timeSeriesLength);

        DataSet ds = new DataSet(features, labels, maskArrayInput, maskArrayOutput);

        mln.fit(ds);
    }

    @Test
    public void testRnnTimeStepWithPreprocessor() {

        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder()
                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                        .list()
                                        .layer(0, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(10)
                                                        .nOut(10).activation(Activation.TANH).build())
                                        .layer(1, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(10)
                                                        .nOut(10).activation(Activation.TANH).build())
                                        .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                        .activation(Activation.SOFTMAX).nIn(10).nOut(10).build())
                                        .inputPreProcessor(0, new FeedForwardToRnnPreProcessor()).pretrain(false)
                                        .backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray in = Nd4j.rand(1, 10);
        net.rnnTimeStep(in);
    }

    @Test
    public void testRnnTimeStepWithPreprocessorGraph() {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .graphBuilder().addInputs("in")
                        .addLayer("0", new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(10).nOut(10)
                                        .activation(Activation.TANH).build(), "in")
                        .addLayer("1", new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(10).nOut(10)
                                        .activation(Activation.TANH).build(), "0")
                        .addLayer("2", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nIn(10).nOut(10).build(), "1")
                        .setOutputs("2").inputPreProcessor("0", new FeedForwardToRnnPreProcessor()).pretrain(false)
                        .backprop(true).build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        INDArray in = Nd4j.rand(1, 10);
        net.rnnTimeStep(in);
    }


    @Test
    public void testTBPTTLongerThanTS() {
        //Extremely simple test of the 'does it throw an exception' variety
        int timeSeriesLength = 20;
        int tbpttLength = 1000;
        int miniBatchSize = 7;
        int nIn = 5;
        int nOut = 4;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.XAVIER).list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(nIn).nOut(7)
                                        .activation(Activation.TANH).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(7).nOut(8)
                                        .activation(Activation.TANH).build())
                        .layer(2, new RnnOutputLayer.Builder(LossFunction.MSE).nIn(8).nOut(nOut)
                                        .activation(Activation.IDENTITY).build())
                        .pretrain(false).backprop(true).backpropType(BackpropType.TruncatedBPTT)
                        .tBPTTBackwardLength(tbpttLength).tBPTTForwardLength(tbpttLength).build();

        Nd4j.getRandom().setSeed(12345);
        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();

        INDArray features = Nd4j.rand(new int[] {miniBatchSize, nIn, timeSeriesLength});
        INDArray labels = Nd4j.rand(new int[] {miniBatchSize, nOut, timeSeriesLength});

        INDArray maskArrayInput = Nd4j.ones(miniBatchSize, timeSeriesLength);
        INDArray maskArrayOutput = Nd4j.ones(miniBatchSize, timeSeriesLength);

        DataSet ds = new DataSet(features, labels, maskArrayInput, maskArrayOutput);

        INDArray initialParams = mln.params().dup();
        mln.fit(ds);
        INDArray afterParams = mln.params();
        assertNotEquals(initialParams, afterParams);
    }
}
