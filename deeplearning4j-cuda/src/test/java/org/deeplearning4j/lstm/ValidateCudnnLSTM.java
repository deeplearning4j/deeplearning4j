package org.deeplearning4j.lstm;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.recurrent.CudnnLSTMHelper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.LSTMParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 18/07/2017.
 */
public class ValidateCudnnLSTM {

    @Test
    public void validateImplSimple() throws Exception {

        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int inputSize = 3;
        int lstmLayerSize = 4;
        int timeSeriesLength = 3;
        int nOut = 2;
        INDArray input = Nd4j.rand(new int[]{minibatch, inputSize, timeSeriesLength});
        INDArray labels = Nd4j.zeros(minibatch, nOut, timeSeriesLength);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            for (int j = 0; j < timeSeriesLength; j++) {
                labels.putScalar(i, r.nextInt(nOut), j, 1.0);
            }
        }

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .inferenceWorkspaceMode(WorkspaceMode.NONE).trainingWorkspaceMode(WorkspaceMode.NONE)
                .learningRate(1.0)
                .regularization(false).updater(Updater.NONE).seed(12345L).weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0, 2)).list()
                .layer(0, new LSTM.Builder().nIn(input.size(1)).nOut(lstmLayerSize)
                        .gateActivationFunction(Activation.SIGMOID).activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(lstmLayerSize).nOut(nOut).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork mln1 = new MultiLayerNetwork(conf.clone());
        mln1.init();

        MultiLayerNetwork mln2 = new MultiLayerNetwork(conf.clone());
        mln2.init();


        assertEquals(mln1.params(), mln2.params());

        Field f = org.deeplearning4j.nn.layers.recurrent.LSTM.class.getDeclaredField("helper");
        f.setAccessible(true);

        Layer l0 = mln1.getLayer(0);
        f.set(l0, null);
        assertNull(f.get(l0));

        l0 = mln2.getLayer(0);
        assertTrue(f.get(l0) instanceof CudnnLSTMHelper);


        INDArray out1 = mln1.output(input);
        INDArray out2 = mln2.output(input);

        assertEquals(out1, out2);


        mln1.setInput(input);
        mln1.setLabels(labels);

        mln2.setInput(input);
        mln2.setLabels(labels);

        mln1.computeGradientAndScore();
        mln2.computeGradientAndScore();

        assertEquals(mln1.score(), mln2.score(), 1e-8);

        Gradient g1 = mln1.gradient();
        Gradient g2 = mln2.gradient();

        for (Map.Entry<String, INDArray> entry : g1.gradientForVariable().entrySet()) {
            INDArray exp = entry.getValue();
            INDArray act = g2.gradientForVariable().get(entry.getKey());

            System.out.println(entry.getKey() + "\t" + exp.equals(act));
        }

        assertEquals(mln1.getFlattenedGradients(), mln2.getFlattenedGradients());
    }

    @Test
    public void validateImplMultiLayer() throws Exception {

        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int inputSize = 3;
        int lstmLayerSize = 4;
        int timeSeriesLength = 3;
        int nOut = 2;
        INDArray input = Nd4j.rand(new int[]{minibatch, inputSize, timeSeriesLength});
        INDArray labels = Nd4j.zeros(minibatch, nOut, timeSeriesLength);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            for (int j = 0; j < timeSeriesLength; j++) {
                labels.putScalar(i, r.nextInt(nOut), j, 1.0);
            }
        }

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(1.0)
                .inferenceWorkspaceMode(WorkspaceMode.NONE).trainingWorkspaceMode(WorkspaceMode.NONE)
                .regularization(false).updater(Updater.NONE).seed(12345L).weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0, 2)).list()
                .layer(0, new LSTM.Builder().nIn(input.size(1)).nOut(lstmLayerSize)
                        .gateActivationFunction(Activation.SIGMOID).activation(Activation.TANH).build())
                .layer(1, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                        .gateActivationFunction(Activation.SIGMOID).activation(Activation.TANH).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(lstmLayerSize).nOut(nOut).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork mln1 = new MultiLayerNetwork(conf.clone());
        mln1.init();

        MultiLayerNetwork mln2 = new MultiLayerNetwork(conf.clone());
        mln2.init();


        assertEquals(mln1.params(), mln2.params());

        Field f = org.deeplearning4j.nn.layers.recurrent.LSTM.class.getDeclaredField("helper");
        f.setAccessible(true);

        Layer l0 = mln1.getLayer(0);
        Layer l1 = mln1.getLayer(1);
        f.set(l0, null);
        f.set(l1, null);
        assertNull(f.get(l0));
        assertNull(f.get(l1));

        l0 = mln2.getLayer(0);
        l1 = mln2.getLayer(1);
        assertTrue(f.get(l0) instanceof CudnnLSTMHelper);
        assertTrue(f.get(l1) instanceof CudnnLSTMHelper);


        INDArray out1 = mln1.output(input);
        INDArray out2 = mln2.output(input);

        assertEquals(out1, out2);


        mln1.setInput(input);
        mln1.setLabels(labels);

        mln2.setInput(input);
        mln2.setLabels(labels);

        mln1.computeGradientAndScore();
        mln2.computeGradientAndScore();

        assertEquals(mln1.getFlattenedGradients(), mln2.getFlattenedGradients());
    }


    @Test
    public void validateImplSimpleDEBUG() throws Exception {

        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int inputSize = 4;
        int lstmLayerSize = 4;
        int timeSeriesLength = 3;
        int nOut = 2;
        INDArray input = Nd4j.rand(new int[]{minibatch, inputSize, timeSeriesLength});
        INDArray labels = Nd4j.zeros(minibatch, nOut, timeSeriesLength);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            for (int j = 0; j < timeSeriesLength; j++) {
                labels.putScalar(i, r.nextInt(nOut), j, 1.0);
            }
        }

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .inferenceWorkspaceMode(WorkspaceMode.NONE).trainingWorkspaceMode(WorkspaceMode.NONE)
                .learningRate(1.0)
                .regularization(false).updater(Updater.NONE).seed(12345L).weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0, 2)).list()
                .layer(0, new LSTM.Builder().nIn(input.size(1)).nOut(lstmLayerSize)
                        .gateActivationFunction(Activation.SIGMOID).activation(Activation.TANH).build())
                .layer(1, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                        .gateActivationFunction(Activation.SIGMOID).activation(Activation.TANH).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(lstmLayerSize).nOut(nOut).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork mln1 = new MultiLayerNetwork(conf.clone());
        mln1.init();

        MultiLayerNetwork mln2 = new MultiLayerNetwork(conf.clone());
        mln2.init();


        //Set all but one set of the params to 0s
        for (int i = 0; i <= 1; i++) {

            Field f = org.deeplearning4j.nn.layers.recurrent.LSTM.class.getDeclaredField("helper");
            f.setAccessible(true);

            Layer l = mln1.getLayer(i);
            f.set(l, null);
            assertNull(f.get(l));

            l = mln2.getLayer(i);
            assertTrue(f.get(l) instanceof CudnnLSTMHelper);

            org.deeplearning4j.nn.layers.recurrent.LSTM lstm = (org.deeplearning4j.nn.layers.recurrent.LSTM) mln1.getLayer(i);

            Map<String, INDArray> paramTable = lstm.paramTable();

            INDArray iw = paramTable.get(LSTMParamInitializer.INPUT_WEIGHT_KEY);
            INDArray rw = paramTable.get(LSTMParamInitializer.RECURRENT_WEIGHT_KEY);
            INDArray b = paramTable.get(LSTMParamInitializer.BIAS_KEY);

//            iw.assign(0);
//            rw.assign(0);
//            b.assign(0);

            int layerNin = iw.size(0);
            int layerSize = iw.size(1) / 4;

            INDArray iGateIW = iw.get(NDArrayIndex.all(), NDArrayIndex.interval(0, layerSize));
            INDArray fGateIW = iw.get(NDArrayIndex.all(), NDArrayIndex.interval(layerSize, 2 * layerSize));
            INDArray oGateIW = iw.get(NDArrayIndex.all(), NDArrayIndex.interval(2 * layerSize, 3 * layerSize));
            INDArray gGateIW = iw.get(NDArrayIndex.all(), NDArrayIndex.interval(3 * layerSize, 4 * layerSize));

//            iGateIW.assign(0);
//            fGateIW.assign(0);
//            oGateIW.assign(0);
//            gGateIW.assign(0);

//            iGateIW.assign(Nd4j.eye(layerSize));

        }

        mln2.setParameters(mln1.params().dup());

        assertEquals(mln1.params(), mln2.params());


        INDArray out1 = mln1.output(input);
        INDArray out2 = mln2.output(input);

        assertEquals(out1, out2);


        mln1.setInput(input);
        mln1.setLabels(labels);

        mln2.setInput(input);
        mln2.setLabels(labels);

        mln1.computeGradientAndScore();
        mln2.computeGradientAndScore();

        assertEquals(mln1.score(), mln2.score(), 1e-8);


        System.out.println("FORWARD PASS: OK");

        Gradient g1 = mln1.gradient();
        Gradient g2 = mln2.gradient();

        System.out.println("--- Gradient correctness by type ---");
        for (Map.Entry<String, INDArray> entry : g1.gradientForVariable().entrySet()) {
            INDArray exp = entry.getValue();
            INDArray act = g2.gradientForVariable().get(entry.getKey());

            System.out.println(entry.getKey() + "\t" + exp.equals(act));

            if(!entry.getKey().endsWith("b") && !entry.getKey().startsWith("2")){

                int layerSize = act.size(1) / 4;

                INDArray i = act.get(NDArrayIndex.all(), NDArrayIndex.interval(0, layerSize));
                INDArray f = act.get(NDArrayIndex.all(), NDArrayIndex.interval(layerSize, 2 * layerSize));
                INDArray o = act.get(NDArrayIndex.all(), NDArrayIndex.interval(2 * layerSize, 3 * layerSize));
                INDArray g = act.get(NDArrayIndex.all(), NDArrayIndex.interval(3 * layerSize, 4 * layerSize));

                INDArray iE = exp.get(NDArrayIndex.all(), NDArrayIndex.interval(0, layerSize));
                INDArray fE = exp.get(NDArrayIndex.all(), NDArrayIndex.interval(layerSize, 2 * layerSize));
                INDArray oE = exp.get(NDArrayIndex.all(), NDArrayIndex.interval(2 * layerSize, 3 * layerSize));
                INDArray gE = exp.get(NDArrayIndex.all(), NDArrayIndex.interval(3 * layerSize, 4 * layerSize));

                System.out.println(entry.getKey()+"-i\t" + iE.equals(i));
                System.out.println(entry.getKey()+"-f\t" + fE.equals(f));
                System.out.println(entry.getKey()+"-o\t" + oE.equals(o));
                System.out.println(entry.getKey()+"-g\t" + gE.equals(g));
            }
        }

        for (int i = 0; i <= 1; i++) {
            INDArray iwg = g1.gradientForVariable().get(i + "_" + LSTMParamInitializer.INPUT_WEIGHT_KEY);
            INDArray rwg = g1.gradientForVariable().get(i + "_" + LSTMParamInitializer.RECURRENT_WEIGHT_KEY);
            INDArray bg = g1.gradientForVariable().get(i + "_" + LSTMParamInitializer.BIAS_KEY);

            INDArray iwgExp = g2.gradientForVariable().get(i + "_" + LSTMParamInitializer.INPUT_WEIGHT_KEY);
            INDArray rwgExp = g2.gradientForVariable().get(i + "_" + LSTMParamInitializer.RECURRENT_WEIGHT_KEY);
            INDArray bgExp = g2.gradientForVariable().get(i + "_" + LSTMParamInitializer.BIAS_KEY);

            System.out.println("Layer gradients by type: " + i);
            System.out.println("iwg");
            System.out.println(iwg.shapeInfoToString());
            System.out.println(iwgExp.shapeInfoToString());
            System.out.println(Arrays.toString(iwg.dup(iwg.ordering()).data().asFloat()));
            System.out.println(Arrays.toString(iwgExp.dup(iwgExp.ordering()).data().asFloat()));
            System.out.println(iwg);
            System.out.println(iwgExp);
            System.out.println("rwg");
            System.out.println(rwg);
            System.out.println(rwgExp);
            System.out.println("bg");
            System.out.println(bg);
            System.out.println(bgExp);
        }


        assertEquals(mln1.getFlattenedGradients(), mln2.getFlattenedGradients());

        System.out.println(Arrays.toString(mln1.getFlattenedGradients().data().asFloat()));
    }

}
