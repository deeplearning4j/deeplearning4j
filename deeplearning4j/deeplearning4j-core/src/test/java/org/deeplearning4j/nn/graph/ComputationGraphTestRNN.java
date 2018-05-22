package org.deeplearning4j.nn.graph;

import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorMultiDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer;
import org.deeplearning4j.nn.layers.recurrent.GravesLSTM;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collections;
import java.util.Map;

import static org.junit.Assert.*;

public class ComputationGraphTestRNN extends BaseDL4JTest {

    @Test
    public void testRnnTimeStepGravesLSTM() {
        Nd4j.getRandom().setSeed(12345);
        int timeSeriesLength = 12;

        //4 layer network: 2 GravesLSTM + DenseLayer + RnnOutputLayer. Hence also tests preprocessors.
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).graphBuilder()
                        .addInputs("in")
                        .addLayer("0", new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(5).nOut(7)
                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build(), "in")
                        .addLayer("1", new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(7).nOut(8)
                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build(), "0")
                        .addLayer("2", new DenseLayer.Builder().nIn(8).nOut(9).activation(Activation.TANH)
                                        .weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0,
                                                        0.5))
                                        .build(), "1")
                        .addLayer("3", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .weightInit(WeightInit.DISTRIBUTION).nIn(9).nOut(4)
                                        .activation(Activation.SOFTMAX).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build(), "2")
                        .setOutputs("3").inputPreProcessor("2", new RnnToFeedForwardPreProcessor())
                        .inputPreProcessor("3", new FeedForwardToRnnPreProcessor()).pretrain(false).backprop(true)
                        .build();
        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        INDArray input = Nd4j.rand(new int[] {3, 5, timeSeriesLength});

        Map<String, INDArray> allOutputActivations = graph.feedForward(input, true);
        INDArray fullOutL0 = allOutputActivations.get("0");
        INDArray fullOutL1 = allOutputActivations.get("1");
        INDArray fullOutL3 = allOutputActivations.get("3");

        assertArrayEquals(new long[] {3, 7, timeSeriesLength}, fullOutL0.shape());
        assertArrayEquals(new long[] {3, 8, timeSeriesLength}, fullOutL1.shape());
        assertArrayEquals(new long[] {3, 4, timeSeriesLength}, fullOutL3.shape());

        int[] inputLengths = {1, 2, 3, 4, 6, 12};

        //Do steps of length 1, then of length 2, ..., 12
        //Should get the same result regardless of step size; should be identical to standard forward pass
        for (int i = 0; i < inputLengths.length; i++) {
            int inLength = inputLengths[i];
            int nSteps = timeSeriesLength / inLength; //each of length inLength

            graph.rnnClearPreviousState();

            for (int j = 0; j < nSteps; j++) {
                int startTimeRange = j * inLength;
                int endTimeRange = startTimeRange + inLength;

                INDArray inputSubset = input.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(startTimeRange, endTimeRange));
                if (inLength > 1)
                    assertTrue(inputSubset.size(2) == inLength);

                INDArray[] outArr = graph.rnnTimeStep(inputSubset);
                assertEquals(1, outArr.length);
                INDArray out = outArr[0];

                INDArray expOutSubset;
                if (inLength == 1) {
                    val sizes = new long[] {fullOutL3.size(0), fullOutL3.size(1), 1};
                    expOutSubset = Nd4j.create(sizes);
                    expOutSubset.tensorAlongDimension(0, 1, 0).assign(fullOutL3.get(NDArrayIndex.all(),
                                    NDArrayIndex.all(), NDArrayIndex.point(startTimeRange)));
                } else {
                    expOutSubset = fullOutL3.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.interval(startTimeRange, endTimeRange));
                }

                assertEquals(expOutSubset, out);

                Map<String, INDArray> currL0State = graph.rnnGetPreviousState("0");
                Map<String, INDArray> currL1State = graph.rnnGetPreviousState("1");

                INDArray lastActL0 = currL0State.get(GravesLSTM.STATE_KEY_PREV_ACTIVATION);
                INDArray lastActL1 = currL1State.get(GravesLSTM.STATE_KEY_PREV_ACTIVATION);

                INDArray expLastActL0 = fullOutL0.tensorAlongDimension(endTimeRange - 1, 1, 0);
                INDArray expLastActL1 = fullOutL1.tensorAlongDimension(endTimeRange - 1, 1, 0);

                assertEquals(expLastActL0, lastActL0);
                assertEquals(expLastActL1, lastActL1);
            }
        }
    }

    @Test
    public void testRnnTimeStep2dInput() {
        Nd4j.getRandom().setSeed(12345);
        int timeSeriesLength = 6;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                        .addLayer("0", new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(5).nOut(7)
                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build(), "in")
                        .addLayer("1", new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(7).nOut(8)
                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0,
                                                        0.5))
                                        .build(), "0")
                        .addLayer("2", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .weightInit(WeightInit.DISTRIBUTION).nIn(8).nOut(4)
                                        .activation(Activation.SOFTMAX).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build(), "1")
                        .setOutputs("2").build();
        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        INDArray input3d = Nd4j.rand(new int[] {3, 5, timeSeriesLength});
        INDArray out3d = graph.rnnTimeStep(input3d)[0];
        assertArrayEquals(out3d.shape(), new long[] {3, 4, timeSeriesLength});

        graph.rnnClearPreviousState();
        for (int i = 0; i < timeSeriesLength; i++) {
            INDArray input2d = input3d.tensorAlongDimension(i, 1, 0);
            INDArray out2d = graph.rnnTimeStep(input2d)[0];

            assertArrayEquals(out2d.shape(), new long[] {3, 4});

            INDArray expOut2d = out3d.tensorAlongDimension(i, 1, 0);
            assertEquals(out2d, expOut2d);
        }

        //Check same but for input of size [3,5,1]. Expect [3,4,1] out
        graph.rnnClearPreviousState();
        for (int i = 0; i < timeSeriesLength; i++) {
            INDArray temp = Nd4j.create(new int[] {3, 5, 1});
            temp.tensorAlongDimension(0, 1, 0).assign(input3d.tensorAlongDimension(i, 1, 0));
            INDArray out3dSlice = graph.rnnTimeStep(temp)[0];
            assertArrayEquals(out3dSlice.shape(), new long[] {3, 4, 1});

            assertTrue(out3dSlice.tensorAlongDimension(0, 1, 0).equals(out3d.tensorAlongDimension(i, 1, 0)));
        }
    }


    @Test
    public void testRnnTimeStepMultipleInOut() {
        //Test rnnTimeStep functionality with multiple inputs and outputs...

        Nd4j.getRandom().setSeed(12345);
        int timeSeriesLength = 12;

        //4 layer network: 2 GravesLSTM + DenseLayer + RnnOutputLayer. Hence also tests preprocessors.
        //Network architecture: lstm0 -> Dense -> RnnOutputLayer0
        // and lstm1 -> Dense -> RnnOutputLayer1
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).graphBuilder()
                        .addInputs("in0", "in1")
                        .addLayer("lstm0",
                                        new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(5).nOut(6)
                                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                                        .dist(new NormalDistribution(0, 0.5)).build(),
                                        "in0")
                        .addLayer("lstm1",
                                        new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(4).nOut(5)
                                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                                        .dist(new NormalDistribution(0, 0.5)).build(),
                                        "in1")
                        .addLayer("dense", new DenseLayer.Builder().nIn(6 + 5).nOut(9).activation(Activation.TANH)
                                        .weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0,
                                                        0.5))
                                        .build(), "lstm0", "lstm1")
                        .addLayer("out0", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .weightInit(WeightInit.DISTRIBUTION).nIn(9).nOut(3)
                                        .activation(Activation.SOFTMAX).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0,
                                                        0.5))
                                        .build(), "dense")
                        .addLayer("out1", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .weightInit(WeightInit.DISTRIBUTION).nIn(9).nOut(4)
                                        .activation(Activation.SOFTMAX).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build(), "dense")
                        .setOutputs("out0", "out1").inputPreProcessor("dense", new RnnToFeedForwardPreProcessor())
                        .inputPreProcessor("out0", new FeedForwardToRnnPreProcessor())
                        .inputPreProcessor("out1", new FeedForwardToRnnPreProcessor()).pretrain(false).backprop(true)
                        .build();
        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        INDArray input0 = Nd4j.rand(new int[] {3, 5, timeSeriesLength});
        INDArray input1 = Nd4j.rand(new int[] {3, 4, timeSeriesLength});

        Map<String, INDArray> allOutputActivations = graph.feedForward(new INDArray[] {input0, input1}, true);
        INDArray fullActLSTM0 = allOutputActivations.get("lstm0");
        INDArray fullActLSTM1 = allOutputActivations.get("lstm1");
        INDArray fullActOut0 = allOutputActivations.get("out0");
        INDArray fullActOut1 = allOutputActivations.get("out1");

        assertArrayEquals(new long[] {3, 6, timeSeriesLength}, fullActLSTM0.shape());
        assertArrayEquals(new long[] {3, 5, timeSeriesLength}, fullActLSTM1.shape());
        assertArrayEquals(new long[] {3, 3, timeSeriesLength}, fullActOut0.shape());
        assertArrayEquals(new long[] {3, 4, timeSeriesLength}, fullActOut1.shape());

        int[] inputLengths = {1, 2, 3, 4, 6, 12};

        //Do steps of length 1, then of length 2, ..., 12
        //Should get the same result regardless of step size; should be identical to standard forward pass
        for (int i = 0; i < inputLengths.length; i++) {
            int inLength = inputLengths[i];
            int nSteps = timeSeriesLength / inLength; //each of length inLength

            graph.rnnClearPreviousState();

            for (int j = 0; j < nSteps; j++) {
                int startTimeRange = j * inLength;
                int endTimeRange = startTimeRange + inLength;

                INDArray inputSubset0 = input0.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(startTimeRange, endTimeRange));
                if (inLength > 1)
                    assertTrue(inputSubset0.size(2) == inLength);

                INDArray inputSubset1 = input1.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(startTimeRange, endTimeRange));
                if (inLength > 1)
                    assertTrue(inputSubset1.size(2) == inLength);

                INDArray[] outArr = graph.rnnTimeStep(inputSubset0, inputSubset1);
                assertEquals(2, outArr.length);
                INDArray out0 = outArr[0];
                INDArray out1 = outArr[1];

                INDArray expOutSubset0;
                if (inLength == 1) {
                    val sizes = new long[] {fullActOut0.size(0), fullActOut0.size(1), 1};
                    expOutSubset0 = Nd4j.create(sizes);
                    expOutSubset0.tensorAlongDimension(0, 1, 0).assign(fullActOut0.get(NDArrayIndex.all(),
                                    NDArrayIndex.all(), NDArrayIndex.point(startTimeRange)));
                } else {
                    expOutSubset0 = fullActOut0.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.interval(startTimeRange, endTimeRange));
                }

                INDArray expOutSubset1;
                if (inLength == 1) {
                    val sizes = new long[] {fullActOut1.size(0), fullActOut1.size(1), 1};
                    expOutSubset1 = Nd4j.create(sizes);
                    expOutSubset1.tensorAlongDimension(0, 1, 0).assign(fullActOut1.get(NDArrayIndex.all(),
                                    NDArrayIndex.all(), NDArrayIndex.point(startTimeRange)));
                } else {
                    expOutSubset1 = fullActOut1.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.interval(startTimeRange, endTimeRange));
                }

                assertEquals(expOutSubset0, out0);
                assertEquals(expOutSubset1, out1);

                Map<String, INDArray> currLSTM0State = graph.rnnGetPreviousState("lstm0");
                Map<String, INDArray> currLSTM1State = graph.rnnGetPreviousState("lstm1");

                INDArray lastActL0 = currLSTM0State.get(GravesLSTM.STATE_KEY_PREV_ACTIVATION);
                INDArray lastActL1 = currLSTM1State.get(GravesLSTM.STATE_KEY_PREV_ACTIVATION);

                INDArray expLastActL0 = fullActLSTM0.tensorAlongDimension(endTimeRange - 1, 1, 0);
                INDArray expLastActL1 = fullActLSTM1.tensorAlongDimension(endTimeRange - 1, 1, 0);

                assertEquals(expLastActL0, lastActL0);
                assertEquals(expLastActL1, lastActL1);
            }
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

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                .trainingWorkspaceMode(WorkspaceMode.NONE).inferenceWorkspaceMode(WorkspaceMode.NONE)
                .graphBuilder()
                        .addInputs("in")
                        .addLayer("0", new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(nIn).nOut(7)
                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build(), "in")
                        .addLayer("1", new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(7).nOut(8)
                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0,
                                                        0.5))
                                        .build(), "0")
                        .addLayer("out", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .weightInit(WeightInit.DISTRIBUTION).nIn(8).nOut(nOut)
                                        .activation(Activation.SOFTMAX).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build(), "1")
                        .setOutputs("out").backprop(true).build();
        assertEquals(BackpropType.Standard, conf.getBackpropType());

        ComputationGraphConfiguration confTBPTT = new NeuralNetConfiguration.Builder().seed(12345)
                .trainingWorkspaceMode(WorkspaceMode.NONE).inferenceWorkspaceMode(WorkspaceMode.NONE)
                .graphBuilder()
                        .addInputs("in")
                        .addLayer("0", new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(nIn).nOut(7)
                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build(), "in")
                        .addLayer("1", new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(7).nOut(8)
                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0,
                                                        0.5))
                                        .build(), "0")
                        .addLayer("out", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .weightInit(WeightInit.DISTRIBUTION).nIn(8).nOut(nOut)
                                        .activation(Activation.SOFTMAX).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build(), "1")
                        .setOutputs("out").backprop(true).backpropType(BackpropType.TruncatedBPTT)
                        .tBPTTForwardLength(timeSeriesLength).tBPTTBackwardLength(timeSeriesLength).build();
        assertEquals(BackpropType.TruncatedBPTT, confTBPTT.getBackpropType());

        Nd4j.getRandom().setSeed(12345);
        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        Nd4j.getRandom().setSeed(12345);
        ComputationGraph graphTBPTT = new ComputationGraph(confTBPTT);
        graphTBPTT.init();
        graphTBPTT.clearTbpttState = false;

        assertEquals(BackpropType.TruncatedBPTT, graphTBPTT.getConfiguration().getBackpropType());
        assertEquals(timeSeriesLength, graphTBPTT.getConfiguration().getTbpttFwdLength());
        assertEquals(timeSeriesLength, graphTBPTT.getConfiguration().getTbpttBackLength());

        INDArray inputData = Nd4j.rand(new int[] {miniBatchSize, nIn, timeSeriesLength});
        INDArray labels = Nd4j.rand(new int[] {miniBatchSize, nOut, timeSeriesLength});

        graph.setInput(0, inputData);
        graph.setLabel(0, labels);

        graphTBPTT.setInput(0, inputData);
        graphTBPTT.setLabel(0, labels);

        graph.computeGradientAndScore();
        graphTBPTT.computeGradientAndScore();

        Pair<Gradient, Double> graphPair = graph.gradientAndScore();
        Pair<Gradient, Double> graphTbpttPair = graphTBPTT.gradientAndScore();

        assertEquals(graphPair.getFirst().gradientForVariable(), graphTbpttPair.getFirst().gradientForVariable());
        assertEquals(graphPair.getSecond(), graphTbpttPair.getSecond());

        //Check states: expect stateMap to be empty but tBpttStateMap to not be
        Map<String, INDArray> l0StateMLN = graph.rnnGetPreviousState(0);
        Map<String, INDArray> l0StateTBPTT = graphTBPTT.rnnGetPreviousState(0);
        Map<String, INDArray> l1StateMLN = graph.rnnGetPreviousState(0);
        Map<String, INDArray> l1StateTBPTT = graphTBPTT.rnnGetPreviousState(0);

        Map<String, INDArray> l0TBPTTState = ((BaseRecurrentLayer<?>) graph.getLayer(0)).rnnGetTBPTTState();
        Map<String, INDArray> l0TBPTTStateTBPTT = ((BaseRecurrentLayer<?>) graphTBPTT.getLayer(0)).rnnGetTBPTTState();
        Map<String, INDArray> l1TBPTTState = ((BaseRecurrentLayer<?>) graph.getLayer(1)).rnnGetTBPTTState();
        Map<String, INDArray> l1TBPTTStateTBPTT = ((BaseRecurrentLayer<?>) graphTBPTT.getLayer(1)).rnnGetTBPTTState();

        assertTrue(l0StateMLN.isEmpty());
        assertTrue(l0StateTBPTT.isEmpty());
        assertTrue(l1StateMLN.isEmpty());
        assertTrue(l1StateTBPTT.isEmpty());

        assertTrue(l0TBPTTState.isEmpty());
        assertEquals(2, l0TBPTTStateTBPTT.size());
        assertTrue(l1TBPTTState.isEmpty());
        assertEquals(2, l1TBPTTStateTBPTT.size());

        INDArray tbpttActL0 = l0TBPTTStateTBPTT.get(GravesLSTM.STATE_KEY_PREV_ACTIVATION);
        INDArray tbpttActL1 = l1TBPTTStateTBPTT.get(GravesLSTM.STATE_KEY_PREV_ACTIVATION);

        Map<String, INDArray> activations = graph.feedForward(inputData, true);
        INDArray l0Act = activations.get("0");
        INDArray l1Act = activations.get("1");
        INDArray expL0Act = l0Act.tensorAlongDimension(timeSeriesLength - 1, 1, 0);
        INDArray expL1Act = l1Act.tensorAlongDimension(timeSeriesLength - 1, 1, 0);
        assertEquals(tbpttActL0, expL0Act);
        assertEquals(tbpttActL1, expL1Act);
    }

    @Test
    public void testTruncatedBPTTSimple() {
        //Extremely simple test of the 'does it throw an exception' variety
        int timeSeriesLength = 12;
        int miniBatchSize = 7;
        int nIn = 5;
        int nOut = 4;

        int nTimeSlices = 20;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).graphBuilder()
                        .addInputs("in")
                        .addLayer("0", new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(nIn).nOut(7)
                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build(), "in")
                        .addLayer("1", new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(7).nOut(8)
                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0,
                                                        0.5))
                                        .build(), "0")
                        .addLayer("out", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .weightInit(WeightInit.DISTRIBUTION).nIn(8).nOut(nOut)
                                        .activation(Activation.SOFTMAX).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build(), "1")
                        .setOutputs("out").pretrain(false).backprop(true).backpropType(BackpropType.TruncatedBPTT)
                        .tBPTTBackwardLength(timeSeriesLength).tBPTTForwardLength(timeSeriesLength).build();

        Nd4j.getRandom().setSeed(12345);
        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        INDArray inputLong = Nd4j.rand(new int[] {miniBatchSize, nIn, nTimeSlices * timeSeriesLength});
        INDArray labelsLong = Nd4j.rand(new int[] {miniBatchSize, nOut, nTimeSlices * timeSeriesLength});

        graph.fit(new INDArray[] {inputLong}, new INDArray[] {labelsLong});
    }

    @Test
    public void testTBPTTLongerThanTS() {
        int tbpttLength = 100;
        int timeSeriesLength = 20;
        int miniBatchSize = 7;
        int nIn = 5;
        int nOut = 4;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).graphBuilder()
                        .addInputs("in")
                        .addLayer("0", new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(nIn).nOut(7)
                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build(), "in")
                        .addLayer("1", new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().nIn(7).nOut(8)
                                        .activation(Activation.TANH).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0,
                                                        0.5))
                                        .build(), "0")
                        .addLayer("out", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .weightInit(WeightInit.DISTRIBUTION).nIn(8).nOut(nOut)
                                        .activation(Activation.SOFTMAX).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 0.5)).build(), "1")
                        .setOutputs("out").pretrain(false).backprop(true).backpropType(BackpropType.TruncatedBPTT)
                        .tBPTTBackwardLength(tbpttLength).tBPTTForwardLength(tbpttLength).build();

        Nd4j.getRandom().setSeed(12345);
        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        INDArray inputLong = Nd4j.rand(new int[] {miniBatchSize, nIn, timeSeriesLength});
        INDArray labelsLong = Nd4j.rand(new int[] {miniBatchSize, nOut, timeSeriesLength});

        INDArray initialParams = graph.params().dup();
        graph.fit(new INDArray[] {inputLong}, new INDArray[] {labelsLong});
        INDArray afterParams = graph.params();

        assertNotEquals(initialParams, afterParams);
    }

    @Test
    public void testTbpttMasking() {
        //Simple "does it throw an exception" type test...
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .graphBuilder().addInputs("in")
                        .addLayer("out", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                        .activation(Activation.IDENTITY).nIn(1).nOut(1).build(), "in")
                        .setOutputs("out").backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(8)
                        .tBPTTBackwardLength(8).build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        MultiDataSet data = new MultiDataSet(new INDArray[] {Nd4j.linspace(1, 10, 10).reshape(1, 1, 10)},
                        new INDArray[] {Nd4j.linspace(2, 20, 10).reshape(1, 1, 10)}, null,
                        new INDArray[] {Nd4j.ones(10)});

        net.fit(data);
    }


    @Test
    public void checkMaskArrayClearance() {
        for (boolean tbptt : new boolean[] {true, false}) {
            //Simple "does it throw an exception" type test...
            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                            .graphBuilder().addInputs("in")
                            .addLayer("out", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                            .activation(Activation.IDENTITY).nIn(1).nOut(1).build(), "in")
                            .setOutputs("out").backpropType(tbptt ? BackpropType.TruncatedBPTT : BackpropType.Standard)
                            .tBPTTForwardLength(8).tBPTTBackwardLength(8).build();

            ComputationGraph net = new ComputationGraph(conf);
            net.init();

            MultiDataSet data = new MultiDataSet(new INDArray[] {Nd4j.linspace(1, 10, 10).reshape(1, 1, 10)},
                            new INDArray[] {Nd4j.linspace(2, 20, 10).reshape(1, 1, 10)}, new INDArray[] {Nd4j.ones(10)},
                            new INDArray[] {Nd4j.ones(10)});

            net.fit(data);
            assertNull(net.getInputMaskArrays());
            assertNull(net.getLabelMaskArrays());
            for (Layer l : net.getLayers()) {
                assertNull(l.getMaskArray());
            }

            DataSet ds = new DataSet(data.getFeatures(0), data.getLabels(0), data.getFeaturesMaskArray(0),
                            data.getLabelsMaskArray(0));
            net.fit(ds);
            assertNull(net.getInputMaskArrays());
            assertNull(net.getLabelMaskArrays());
            for (Layer l : net.getLayers()) {
                assertNull(l.getMaskArray());
            }

            net.fit(data.getFeatures(), data.getLabels(), data.getFeaturesMaskArrays(), data.getLabelsMaskArrays());
            assertNull(net.getInputMaskArrays());
            assertNull(net.getLabelMaskArrays());
            for (Layer l : net.getLayers()) {
                assertNull(l.getMaskArray());
            }

            MultiDataSetIterator iter = new IteratorMultiDataSetIterator(
                            Collections.singletonList((org.nd4j.linalg.dataset.api.MultiDataSet) data).iterator(), 1);
            net.fit(iter);
            assertNull(net.getInputMaskArrays());
            assertNull(net.getLabelMaskArrays());
            for (Layer l : net.getLayers()) {
                assertNull(l.getMaskArray());
            }

            DataSetIterator iter2 = new IteratorDataSetIterator(Collections.singletonList(ds).iterator(), 1);
            net.fit(iter2);
            assertNull(net.getInputMaskArrays());
            assertNull(net.getLabelMaskArrays());
            for (Layer l : net.getLayers()) {
                assertNull(l.getMaskArray());
            }
        }
    }

}
