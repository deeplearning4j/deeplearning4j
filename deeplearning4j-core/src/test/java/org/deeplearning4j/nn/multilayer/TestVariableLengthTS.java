package org.deeplearning4j.nn.multilayer;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class TestVariableLengthTS {

    @Test
    public void testVariableLengthSimple(){

        //Test: Simple RNN layer + RNNOutputLayer
        //Length of 4 for standard
        //Length of 5 with last time step output mask set to 0
        //Expect the same gradients etc in both cases...

        int[] miniBatchSizes = {1,2,5};
        int nOut = 1;
        Random r = new Random(12345);

        for (int nExamples : miniBatchSizes) {
            Nd4j.getRandom().setSeed(12345);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                    .updater(Updater.SGD)
                    .learningRate(0.1)
                    .seed(12345)
                    .list()
                    .layer(0, new GravesLSTM.Builder().activation("tanh").nIn(2).nOut(2).build())
                    .layer(1, new RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(2).nOut(1).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            INDArray in1 = Nd4j.rand(new int[]{nExamples, 2, 4});
            INDArray in2 = Nd4j.rand(new int[]{nExamples, 2, 5});
            in2.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 3, true)}, in1);

            assertEquals(in1, in2.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)));

            INDArray labels1 = Nd4j.rand(new int[]{nExamples, 1, 4});
            INDArray labels2 = Nd4j.create(nExamples, 1, 5);
            labels2.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 3, true)}, labels1);
            assertEquals(labels1, labels2.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)));

            INDArray labelMask = Nd4j.ones(nExamples, 5);
            for (int j = 0; j < nExamples; j++) {
                labelMask.putScalar(new int[]{j, 4}, 0);
            }


            net.setInput(in1);
            net.setLabels(labels1);
            net.computeGradientAndScore();
            double score1 = net.score();
            Gradient g1 = net.gradient();

            net.setInput(in2);
            net.setLabels(labels2);
            net.setLayerMaskArrays(null, labelMask);
            net.computeGradientAndScore();
            double score2 = net.score();
            Gradient g2 = net.gradient();

            //Scores and gradients should be identical for two cases (given mask array)
            assertEquals(score1, score2, 0.0);

            Map<String, INDArray> g1map = g1.gradientForVariable();
            Map<String, INDArray> g2map = g2.gradientForVariable();

            for (String s : g1map.keySet()) {
                INDArray g1s = g1map.get(s);
                INDArray g2s = g2map.get(s);
                assertEquals(s, g1s, g2s);
            }

            //Finally: check that the values at the masked outputs don't actually make any differente to:
            // (a) score, (b) gradients
            for( int i=0; i<nExamples; i++ ){
                for( int j=0; j<nOut; j++ ){
                    double d = r.nextDouble();
                    labels2.putScalar(new int[]{i,j,4},d);
                }
                net.setLabels(labels2);
                net.computeGradientAndScore();
                double score2a = net.score();
                Gradient g2a = net.gradient();
                assertEquals(score2,score2a,0.0);
                for (String s : g2map.keySet()) {
                    INDArray g2s = g2map.get(s);
                    INDArray g2sa = g2a.getGradientFor(s);
                    assertEquals(s, g2s, g2sa);
                }
            }
        }
    }

    @Test
    public void testInputMasking(){
        //Idea: have masking on the input with 2 dense layers on input
        //Ensure that the parameter gradients for the inputs don't depend on the inputs when inputs are masked

        int[] miniBatchSizes = {1,2,5};
        int nIn = 2;
        Random r = new Random(12345);

        for (int nExamples : miniBatchSizes) {
            Nd4j.getRandom().setSeed(12345);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                    .updater(Updater.SGD)
                    .learningRate(0.1)
                    .seed(12345)
                    .list()
                    .layer(0, new DenseLayer.Builder().activation("tanh").nIn(2).nOut(2).build())
                    .layer(1, new DenseLayer.Builder().activation("tanh").nIn(2).nOut(2).build())
                    .layer(2, new GravesLSTM.Builder().activation("tanh").nIn(2).nOut(2).build())
                    .layer(3, new RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(2).nOut(1).build())
                    .inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
                    .inputPreProcessor(2, new FeedForwardToRnnPreProcessor())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            INDArray in1 = Nd4j.rand(new int[]{nExamples, 2, 4});
            INDArray in2 = Nd4j.rand(new int[]{nExamples, 2, 5});
            in2.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 3, true)}, in1);

            assertEquals(in1, in2.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)));

            INDArray labels1 = Nd4j.rand(new int[]{nExamples, 1, 4});
            INDArray labels2 = Nd4j.create(nExamples, 1, 5);
            labels2.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 3, true)}, labels1);
            assertEquals(labels1, labels2.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)));

            INDArray inputMask = Nd4j.ones(nExamples, 5);
            for (int j = 0; j < nExamples; j++) {
                inputMask.putScalar(new int[]{j, 4}, 0);
            }


            net.setInput(in1);
            net.setLabels(labels1);
            net.computeGradientAndScore();
            double score1 = net.score();
            Gradient g1 = net.gradient();

            net.setInput(in2);
            net.setLabels(labels2);
            net.setLayerMaskArrays(inputMask,null);
            net.computeGradientAndScore();
            double score2 = net.score();
            Gradient g2 = net.gradient();
            List<INDArray> activations2 = net.feedForward();

            //Scores should differ here: masking the input, not the output. Therefore 4 vs. 5 time step outputs
            assertNotEquals(score1, score2, 0.01);

            Map<String, INDArray> g1map = g1.gradientForVariable();
            Map<String, INDArray> g2map = g2.gradientForVariable();

            for (String s : g1map.keySet()) {
                INDArray g1s = g1map.get(s);
                INDArray g2s = g2map.get(s);

                assertNotEquals(s, g1s, g2s);
            }

            //Modify the values at the masked time step, and check that neither the gradients, score or activations change
            for( int j=0; j<nExamples; j++ ){
                for( int k=0; k<nIn; k++ ) {
                    in2.putScalar(new int[]{j,k,4},r.nextDouble());
                }
                net.setInput(in2);
                net.computeGradientAndScore();
                double score2a = net.score();
                Gradient g2a = net.gradient();
                assertEquals(score2,score2a,1e-12);
                for(String s : g2.gradientForVariable().keySet()){
                    assertEquals(g2.getGradientFor(s),g2a.getGradientFor(s));
                }

                List<INDArray> activations2a = net.feedForward();
                for( int k=1; k<activations2.size(); k++ ){
                    assertEquals(activations2.get(k),activations2a.get(k));
                }
            }

            //Finally: check that the activations for the first two (dense) layers are zero at the appropriate time step
            FeedForwardToRnnPreProcessor temp = new FeedForwardToRnnPreProcessor();
            INDArray l0Before = activations2.get(1);
            INDArray l1Before = activations2.get(2);
            INDArray l0After = temp.preProcess(l0Before, nExamples);
            INDArray l1After = temp.preProcess(l1Before,nExamples);

            for( int j=0; j<nExamples; j++ ){
                for( int k=0; k<nIn; k++ ) {
                    assertEquals(0.0, l0After.getDouble(j,k,4),0.0);
                    assertEquals(0.0, l1After.getDouble(j,k,4),0.0);
                }
            }
        }
    }

    @Test
    public void testOutputMaskingScoreMagnitudes(){
        //Idea: check magnitude of scores, with differeing number of values masked out
        //i.e., MSE with zero weight init and 1.0 labels: know what to expect in terms of score

        int nIn = 3;
        int[] timeSeriesLengths = {3,10};
        int[] outputSizes = {1,2,5};
        int[] miniBatchSizes = {1,4};

        Random r = new Random(12345);

        for(int tsLength : timeSeriesLengths ){
            for( int nOut : outputSizes ){
                for( int miniBatch : miniBatchSizes ) {
                    for (int nToMask = 0; nToMask < tsLength-1; nToMask++ ) {
                        String msg = "tsLen="+tsLength + ", nOut="+nOut + ", miniBatch="+miniBatch;

                        INDArray labelMaskArray = Nd4j.ones(miniBatch, tsLength);
                        for( int i=0; i<miniBatch; i++ ){
                            //For each example: select which outputs to mask...
                            int nMasked = 0;
                            while(nMasked < nToMask){
                                int tryIdx = r.nextInt(tsLength);
                                if(labelMaskArray.getDouble(i,tryIdx)==0.0) continue;
                                labelMaskArray.putScalar(new int[]{i,tryIdx},0.0);
                                nMasked++;
                            }
                        }

                        INDArray input = Nd4j.rand(new int[]{miniBatch, nIn, tsLength});
                        INDArray labels = Nd4j.ones(miniBatch, nOut, tsLength);

                        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                .regularization(false)
                                .seed(12345L)
                                .list()
                                .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(5).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 1)).updater(Updater.NONE).build())
                                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation("identity").nIn(5).nOut(nOut)
                                        .weightInit(WeightInit.ZERO).updater(Updater.NONE).build())
                                .pretrain(false).backprop(true)
                                .build();
                        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                        mln.init();

                        //MSE loss function: 1/2n * sum(squaredErrors)
                        double expScore = 0.5 * nOut * (tsLength-nToMask);  //Sum over minibatches, then divide by minibatch size

                        mln.setLayerMaskArrays(null, labelMaskArray);
                        mln.setInput(input);
                        mln.setLabels(labels);

                        mln.computeGradientAndScore();
                        double score = mln.score();

                        assertEquals(msg,expScore,score,0.1);
                    }
                }
            }
        }
    }

    @Test
    public void testOutputMasking(){
        //If labels are masked: want zero outputs for that time step.

        int nIn = 3;
        int[] timeSeriesLengths = {3,10};
        int[] outputSizes = {1,2,5};
        int[] miniBatchSizes = {1,4};

        Random r = new Random(12345);

        for(int tsLength : timeSeriesLengths ){
            for( int nOut : outputSizes ){
                for( int miniBatch : miniBatchSizes ) {
                    for (int nToMask = 0; nToMask < tsLength-1; nToMask++ ) {
                        INDArray labelMaskArray = Nd4j.ones(miniBatch, tsLength);
                        for( int i=0; i<miniBatch; i++ ){
                            //For each example: select which outputs to mask...
                            int nMasked = 0;
                            while(nMasked < nToMask){
                                int tryIdx = r.nextInt(tsLength);
                                if(labelMaskArray.getDouble(i,tryIdx)==0.0) continue;
                                labelMaskArray.putScalar(new int[]{i,tryIdx},0.0);
                                nMasked++;
                            }
                        }

                        INDArray input = Nd4j.rand(new int[]{miniBatch, nIn, tsLength});

                        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                .regularization(false)
                                .seed(12345L)
                                .list()
                                .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(5).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 1)).updater(Updater.NONE).build())
                                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation("identity").nIn(5).nOut(nOut)
                                        .weightInit(WeightInit.XAVIER).updater(Updater.NONE).build())
                                .pretrain(false).backprop(true)
                                .build();
                        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                        mln.init();

                        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                                .regularization(false)
                                .seed(12345L)
                                .list()
                                .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(5).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 1)).updater(Updater.NONE).build())
                                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax").nIn(5).nOut(nOut)
                                        .weightInit(WeightInit.XAVIER).updater(Updater.NONE).build())
                                .pretrain(false).backprop(true)
                                .build();
                        MultiLayerNetwork mln2 = new MultiLayerNetwork(conf2);
                        mln2.init();

                        mln.setLayerMaskArrays(null, labelMaskArray);
                        mln2.setLayerMaskArrays(null, labelMaskArray);


                        INDArray out = mln.output(input);
                        INDArray out2 = mln2.output(input);
                        for( int i=0; i<miniBatch; i++ ){
                            for( int j=0; j<tsLength; j++ ){
                                double m = labelMaskArray.getDouble(i,j);
                                if(m == 0.0){
                                    //Expect outputs to be exactly 0.0
                                    INDArray outRow = out.get(NDArrayIndex.point(i),NDArrayIndex.all(),NDArrayIndex.point(j));
                                    INDArray outRow2 = out2.get(NDArrayIndex.point(i),NDArrayIndex.all(),NDArrayIndex.point(j));
                                    for( int k=0; k<nOut; k++ ){
                                        assertEquals(outRow.getDouble(k),0.0,0.0);
                                        assertEquals(outRow2.getDouble(k),0.0,0.0);
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
