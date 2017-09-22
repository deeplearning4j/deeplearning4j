package org.deeplearning4j.nn.graph;

import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class TestVariableLengthTSCG {

    private static final ActivationsFactory af = ActivationsFactory.getInstance();

    @Test
    public void testVariableLengthSimple() {

        //Test: Simple RNN layer + RNNOutputLayer
        //Length of 4 for standard
        //Length of 5 with last time step output mask set to 0
        //Expect the same gradients etc in both cases...

        int[] miniBatchSizes = {1, 2, 5};
        int nOut = 1;
        Random r = new Random(12345);

        for (int nExamples : miniBatchSizes) {
            Nd4j.getRandom().setSeed(12345);

            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                            .updater(new Sgd(0.1)).seed(12345).graphBuilder().addInputs("in")
                            .addLayer("0", new GravesLSTM.Builder().activation(Activation.TANH).nIn(2).nOut(2).build(),
                                            "in")
                            .addLayer("1", new RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE)
                                            .nIn(2).nOut(1).build(), "0")
                            .setOutputs("1").build();

            ComputationGraph net = new ComputationGraph(conf);
            net.init();

            INDArray in1 = Nd4j.rand(new int[] {nExamples, 2, 4});
            INDArray in2 = Nd4j.rand(new int[] {nExamples, 2, 5});
            in2.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 3, true)},
                            in1);

            assertEquals(in1, in2.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)));

            INDArray labels1 = Nd4j.rand(new int[] {nExamples, 1, 4});
            INDArray labels2 = Nd4j.create(nExamples, 1, 5);
            labels2.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 3, true)},
                            labels1);
            assertEquals(labels1, labels2.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)));

            INDArray labelMask = Nd4j.ones(nExamples, 5);
            for (int j = 0; j < nExamples; j++) {
                labelMask.putScalar(new int[] {j, 4}, 0);
            }


            Pair<Gradients,Double> p1 = net.computeGradientAndScore(af.create(in1), af.create(labels1));
            double score1 = net.score();
            Gradient g1 = p1.getFirst().getParameterGradients();

            Pair<Gradients,Double> p2 = net.computeGradientAndScore(af.create(in2), af.create(labels2, labelMask));
            double score2 = net.score();
            Gradient g2 = p2.getFirst().getParameterGradients();

            //Scores and gradients should be identical for two cases (given mask array)
            assertEquals(score1, score2, 1e-6);

            Map<String, INDArray> g1map = g1.gradientForVariable();
            Map<String, INDArray> g2map = g2.gradientForVariable();

            for (String s : g1map.keySet()) {
                INDArray g1s = g1map.get(s);
                INDArray g2s = g2map.get(s);
                assertEquals(s, g1s, g2s);
            }

            //Finally: check that the values at the masked outputs don't actually make any difference to:
            // (a) score, (b) gradients
            for (int i = 0; i < nExamples; i++) {
                for (int j = 0; j < nOut; j++) {
                    double d = r.nextDouble();
                    labels2.putScalar(new int[] {i, j, 4}, d);
                }
                p2 = net.computeGradientAndScore(af.create(in2), af.create(labels2, labelMask));
                double score2a = net.score();
                Gradient g2a = p2.getFirst().getParameterGradients();
                assertEquals(score2, score2a, 1e-6);
                for (String s : g2map.keySet()) {
                    INDArray g2s = g2map.get(s);
                    INDArray g2sa = g2a.getGradientFor(s);
                    assertEquals(s, g2s, g2sa);
                }
            }
        }
    }

    @Test
    public void testInputMasking() {
        //Idea: have masking on the input with 2 dense layers on input
        //Ensure that the parameter gradients for the inputs don't depend on the inputs when inputs are masked

        int[] miniBatchSizes = {1, 2, 5};
        int nIn = 2;
        Random r = new Random(12345);

        for (int nExamples : miniBatchSizes) {
            Nd4j.getRandom().setSeed(12345);

            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                            .updater(new Sgd(0.1)).seed(12345).graphBuilder().addInputs("in")
                            .addLayer("0", new DenseLayer.Builder().activation(Activation.TANH).nIn(2).nOut(2).build(),
                                            "in")
                            .addLayer("1", new DenseLayer.Builder().activation(Activation.TANH).nIn(2).nOut(2).build(),
                                            "0")
                            .addLayer("2", new GravesLSTM.Builder().activation(Activation.TANH).nIn(2).nOut(2).build(),
                                            "1")
                            .addLayer("3", new GravesLSTM.Builder().activation(Activation.TANH).nIn(2).nOut(2).build(),
                                    "2")
                            .addLayer("4", new RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE)
                                            .nIn(2).nOut(1).build(), "3")
                            .setOutputs("4")
                            .inputPreProcessor("0", new RnnToFeedForwardPreProcessor())
                            .inputPreProcessor("2", new FeedForwardToRnnPreProcessor())
                            .build();

            ComputationGraph net = new ComputationGraph(conf);
            net.init();

            INDArray in1 = Nd4j.rand(new int[] {nExamples, 2, 4});
            INDArray in2 = Nd4j.rand(new int[] {nExamples, 2, 5});
            in2.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 3, true)},
                            in1);

            assertEquals(in1, in2.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)));

            INDArray labels1 = Nd4j.rand(new int[] {nExamples, 1, 4});
            INDArray labels2 = Nd4j.create(nExamples, 1, 5);
            labels2.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 3, true)},
                            labels1);
            assertEquals(labels1, labels2.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)));

            INDArray inputMask = Nd4j.ones(nExamples, 5);
            for (int j = 0; j < nExamples; j++) {
                inputMask.putScalar(new int[] {j, 4}, 0);
            }

            //Compute score + gradients without input mask:
            Pair<Gradients,Double> p1 = net.computeGradientAndScore(af.create(in1), af.create(labels1));
            double score1 = net.score();
            Gradient g1 = p1.getFirst().getParameterGradients();
            Map<String, INDArray> g1map = new HashMap<>();
            for (String s : g1.gradientForVariable().keySet()) {
                g1map.put(s, g1.gradientForVariable().get(s).dup()); //Gradients are views; need to dup otherwise they will be modified by next computeGradientAndScore
            }


            //Compute score and gradients *with* input mask
            Activations a = ActivationsFactory.getInstance().create(in2, inputMask);
            System.out.println("Net 2");
            Pair<Gradients,Double> p2 = net.computeGradientAndScore(a, af.create(labels2));
            double score2 = net.score();
            Gradient g2 = p2.getFirst().getParameterGradients();
            Map<String, Activations> activations2 = net.feedForward(a);

            //Scores should differ here: masking the input, not the output. Therefore 4 vs. 5 time step outputs
            assertNotEquals(score1, score2, 0.01);

            Map<String, INDArray> g2map = g2.gradientForVariable();

            //Gradients should be identical for the dense layers (given input masking) but must be different for the LSTM and RNN output
            //However: note that the LSTM Input weight gradients (but not recurrent!) on the first LSTM will also be the same - they
            // get 0s an input from the masked steps... consequently, their gradients for the masked step get multiplied
            // by 0
            for (String s : g1map.keySet()) {
                INDArray g1s = g1map.get(s);
                INDArray g2s = g2map.get(s);

//                System.out.println(s);

                if(s.startsWith("0_") || s.startsWith("1_") || s.equals("2_W")){
                    assertEquals(s, g1s, g2s);
                } else if(s.startsWith("2_") || s.startsWith("3_") || s.startsWith("4_")){
                    if(g1s.equals(g2s)){
                        System.out.println(s + "\t - WRONG - should differ");
                    } else {
                        System.out.println(s + "\t - OK");
                    }
//                    assertNotEquals(s, g1s, g2s);
                } else {
                    throw new RuntimeException(s);
                }
            }

            //Modify the values at the masked time step, and check that neither the gradients, score or activations change
            for (int j = 0; j < nExamples; j++) {
                for (int k = 0; k < nIn; k++) {
                    in2.putScalar(new int[] {j, k, 4}, r.nextDouble());
                }
                a = ActivationsFactory.getInstance().create(in2, inputMask);
                Pair<Gradients,Double> p = net.computeGradientAndScore(a, af.create(labels2));
                double score2a = net.score();
                Gradient g2a = p.getFirst().getParameterGradients();
                assertEquals(score2, score2a, 1e-12);
                for (String s : g2.gradientForVariable().keySet()) {
                    assertEquals(g2.getGradientFor(s), g2a.getGradientFor(s));
                }

                Map<String, Activations> activations2a = net.feedForward(a);
                for (String s : activations2.keySet()) {
                    assertEquals(activations2.get(s), activations2a.get(s));
                }
            }

            //Finally: check that the activations for the first two (dense) layers are zero at the appropriate time step
            FeedForwardToRnnPreProcessor temp = new FeedForwardToRnnPreProcessor();
            INDArray l0Before = activations2.get("0").get(0);
            INDArray l1Before = activations2.get("1").get(0);
            INDArray l0After = temp.preProcess(af.create(l0Before), nExamples, true).get(0);
            INDArray l1After = temp.preProcess(af.create(l1Before), nExamples, true).get(0);

            for (int j = 0; j < nExamples; j++) {
                for (int k = 0; k < nIn; k++) {
                    assertEquals(0.0, l0After.getDouble(j, k, 4), 0.0);
                    assertEquals(0.0, l1After.getDouble(j, k, 4), 0.0);
                }
            }
        }
    }

    @Test
    public void testOutputMaskingScoreMagnitudes() {
        //Idea: check magnitude of scores, with differing number of values masked out
        //i.e., MSE with zero weight init and 1.0 labels: know what to expect in terms of score

        int nIn = 3;
        int[] timeSeriesLengths = {3, 10};
        int[] outputSizes = {1, 2, 5};
        int[] miniBatchSizes = {1, 4};

        Random r = new Random(12345);

        for (int tsLength : timeSeriesLengths) {
            for (int nOut : outputSizes) {
                for (int miniBatch : miniBatchSizes) {
                    for (int nToMask = 0; nToMask < tsLength - 1; nToMask++) {
                        String msg = "tsLen=" + tsLength + ", nOut=" + nOut + ", miniBatch=" + miniBatch;

                        INDArray labelMaskArray = Nd4j.ones(miniBatch, tsLength);
                        for (int i = 0; i < miniBatch; i++) {
                            //For each example: select which outputs to mask...
                            int nMasked = 0;
                            while (nMasked < nToMask) {
                                int tryIdx = r.nextInt(tsLength);
                                if (labelMaskArray.getDouble(i, tryIdx) == 0.0)
                                    continue;
                                labelMaskArray.putScalar(new int[] {i, tryIdx}, 0.0);
                                nMasked++;
                            }
                        }

                        INDArray input = Nd4j.rand(new int[] {miniBatch, nIn, tsLength});
                        INDArray labels = Nd4j.ones(miniBatch, nOut, tsLength);

                        ComputationGraphConfiguration conf =
                                        new NeuralNetConfiguration.Builder().seed(12345L)
                                                        .graphBuilder()
                                                        .addInputs("in").addLayer("0",
                                                                        new GravesLSTM.Builder().nIn(nIn).nOut(5)
                                                                                        .weightInit(WeightInit.DISTRIBUTION)
                                                                                        .dist(new NormalDistribution(0,
                                                                                                        1))
                                                                                        .updater(new NoOp()).build(),
                                                                        "in")
                                                        .addLayer("1", new RnnOutputLayer.Builder(
                                                                        LossFunctions.LossFunction.MSE)
                                                                                        .activation(Activation.IDENTITY)
                                                                                        .nIn(5).nOut(nOut)
                                                                                        .weightInit(WeightInit.ZERO)
                                                                                        .updater(new NoOp()).build(),
                                                                        "0")
                                                        .setOutputs("1").pretrain(false).backprop(true).build();
                        ComputationGraph net = new ComputationGraph(conf);
                        net.init();

                        //MSE loss function: 1/n * sum(squaredErrors)... but sum(squaredErrors) = n * (1-0) here -> sum(squaredErrors)
                        double expScore = tsLength - nToMask; //Sum over minibatches, then divide by minibatch size

                        net.computeGradientAndScore(af.create(input), af.create(labels, labelMaskArray));
                        double score = net.score();

                        assertEquals(msg, expScore, score, 0.1);
                    }
                }
            }
        }
    }

    @Test
    public void testOutputMasking() {
        //If labels are masked: want zero outputs for that time step.

        int nIn = 3;
        int[] timeSeriesLengths = {3, 10};
        int[] outputSizes = {1, 2, 5};
        int[] miniBatchSizes = {1, 4};

        Random r = new Random(12345);

        for (int tsLength : timeSeriesLengths) {
            for (int nOut : outputSizes) {
                for (int miniBatch : miniBatchSizes) {
                    for (int nToMask = 0; nToMask < tsLength - 1; nToMask++) {
                        INDArray labelMaskArray = Nd4j.ones(miniBatch, tsLength);
                        for (int i = 0; i < miniBatch; i++) {
                            //For each example: select which outputs to mask...
                            int nMasked = 0;
                            while (nMasked < nToMask) {
                                int tryIdx = r.nextInt(tsLength);
                                if (labelMaskArray.getDouble(i, tryIdx) == 0.0)
                                    continue;
                                labelMaskArray.putScalar(new int[] {i, tryIdx}, 0.0);
                                nMasked++;
                            }
                        }

                        INDArray input = Nd4j.rand(new int[] {miniBatch, nIn, tsLength});

                        ComputationGraphConfiguration conf =
                                        new NeuralNetConfiguration.Builder().seed(12345L)
                                                        .graphBuilder()
                                                        .addInputs("in").addLayer("0",
                                                                        new GravesLSTM.Builder().nIn(nIn).nOut(5)
                                                                                        .weightInit(WeightInit.DISTRIBUTION)
                                                                                        .dist(new NormalDistribution(0,
                                                                                                        1))
                                                                                        .updater(new NoOp()).build(),
                                                                        "in")
                                                        .addLayer("1", new RnnOutputLayer.Builder(
                                                                        LossFunctions.LossFunction.MSE)
                                                                                        .activation(Activation.IDENTITY)
                                                                                        .nIn(5).nOut(nOut)
                                                                                        .weightInit(WeightInit.XAVIER)
                                                                                        .updater(new NoOp()).build(),
                                                                        "0")
                                                        .setOutputs("1").pretrain(false).backprop(true).build();
                        ComputationGraph net = new ComputationGraph(conf);
                        net.init();

                        ComputationGraphConfiguration conf2 =
                                        new NeuralNetConfiguration.Builder().seed(12345L)
                                                        .graphBuilder()
                                                        .addInputs("in").addLayer("0",
                                                                        new GravesLSTM.Builder().nIn(nIn).nOut(5)
                                                                                        .weightInit(WeightInit.DISTRIBUTION)
                                                                                        .dist(new NormalDistribution(0,
                                                                                                        1))
                                                                                        .updater(new NoOp()).build(),
                                                                        "in")
                                                        .addLayer("1", new RnnOutputLayer.Builder(
                                                                        LossFunctions.LossFunction.MCXENT)
                                                                                        .activation(Activation.SOFTMAX)
                                                                                        .nIn(5).nOut(nOut)
                                                                                        .weightInit(WeightInit.XAVIER)
                                                                                        .updater(new NoOp()).build(),
                                                                        "0")
                                                        .setOutputs("1").pretrain(false).backprop(true).build();
                        ComputationGraph net2 = new ComputationGraph(conf2);
                        net2.init();

                        net.setLayerMaskArrays(null, new INDArray[] {labelMaskArray});
                        net2.setLayerMaskArrays(null, new INDArray[] {labelMaskArray});


                        INDArray out = net.output(input).get(0);
                        INDArray out2 = net2.output(input).get(0);
                        for (int i = 0; i < miniBatch; i++) {
                            for (int j = 0; j < tsLength; j++) {
                                double m = labelMaskArray.getDouble(i, j);
                                if (m == 0.0) {
                                    //Expect outputs to be exactly 0.0
                                    INDArray outRow = out.get(NDArrayIndex.point(i), NDArrayIndex.all(),
                                                    NDArrayIndex.point(j));
                                    INDArray outRow2 = out2.get(NDArrayIndex.point(i), NDArrayIndex.all(),
                                                    NDArrayIndex.point(j));
                                    for (int k = 0; k < nOut; k++) {
                                        assertEquals(outRow.getDouble(k), 0.0, 0.0);
                                        assertEquals(outRow2.getDouble(k), 0.0, 0.0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}
