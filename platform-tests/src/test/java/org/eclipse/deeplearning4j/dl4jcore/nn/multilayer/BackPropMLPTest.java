/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.dl4jcore.nn.multilayer;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ListBuilder;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SigmoidDerivative;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.fail;
import org.junit.jupiter.api.DisplayName;

@Slf4j
@DisplayName("Back Prop MLP Test")
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
class BackPropMLPTest extends BaseDL4JTest {

    @Test
    @DisplayName("Test MLP Trivial")
    void testMLPTrivial() {
      Nd4j.getEnvironment().setDeleteShapeInfo(false);
      Nd4j.getEnvironment().setDeletePrimary(false);
      Nd4j.getEnvironment().setDeleteSpecial(false);
        // Simplest possible case: 1 hidden layer, 1 hidden neuron, batch size of 1.
        MultiLayerNetwork network = new MultiLayerNetwork(getIrisMLPSimpleConfig(new int[] { 1 }, Activation.SIGMOID));
        network.setListeners(new ScoreIterationListener(1));
        network.init();
        DataSetIterator iter = new IrisDataSetIterator(1, 10);
        while (iter.hasNext()) network.fit(iter.next());
    }

    @Test
    @DisplayName("Test MLP")
    void testMLP() {
        // Simple mini-batch test with multiple hidden layers
        MultiLayerConfiguration conf = getIrisMLPSimpleConfig(new int[] { 5, 4, 3 }, Activation.SIGMOID);
        // System.out.println(conf);
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        DataSetIterator iter = new IrisDataSetIterator(10, 100);
        while (iter.hasNext()) {
            network.fit(iter.next());
        }
    }

    @Test
    @DisplayName("Test MLP 2")
    void testMLP2() {
        // Simple mini-batch test with multiple hidden layers
        MultiLayerConfiguration conf = getIrisMLPSimpleConfig(new int[] { 5, 15, 3 }, Activation.TANH);
        // System.out.println(conf);
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        DataSetIterator iter = new IrisDataSetIterator(12, 120);
        while (iter.hasNext()) {
            network.fit(iter.next());
        }
    }

    @Test
    @DisplayName("Test Single Example Weight Updates")
    void testSingleExampleWeightUpdates() {
        // Simplest possible case: 1 hidden layer, 1 hidden neuron, batch size of 1.
        // Manually calculate weight updates (entirely outside of DL4J and ND4J)
        // and compare expected and actual weights after backprop
        DataSetIterator iris = new IrisDataSetIterator(1, 10);
        MultiLayerNetwork network = new MultiLayerNetwork(getIrisMLPSimpleConfig(new int[] { 1 }, Activation.SIGMOID));
        network.init();
        Layer[] layers = network.getLayers();
        final boolean printCalculations = false;
        while (iris.hasNext()) {
            DataSet data = iris.next();
            INDArray x = data.getFeatures();
            INDArray y = data.getLabels();
            float[] xFloat = asFloat(x);
            float[] yFloat = asFloat(y);
            // Do forward pass:
            // Hidden layer
            INDArray l1Weights = layers[0].getParam(DefaultParamInitializer.WEIGHT_KEY).dup();
            // Output layer
            INDArray l2Weights = layers[1].getParam(DefaultParamInitializer.WEIGHT_KEY).dup();
            INDArray l1Bias = layers[0].getParam(DefaultParamInitializer.BIAS_KEY).dup();
            INDArray l2Bias = layers[1].getParam(DefaultParamInitializer.BIAS_KEY).dup();
            float[] l1WeightsFloat = asFloat(l1Weights);
            float[] l2WeightsFloat = asFloat(l2Weights);
            float l1BiasFloat = l1Bias.getFloat(0);
            float[] l2BiasFloatArray = asFloat(l2Bias);
            // z=w*x+b
            float hiddenUnitPreSigmoid = dotProduct(l1WeightsFloat, xFloat) + l1BiasFloat;
            // a=sigma(z)
            float hiddenUnitPostSigmoid = sigmoid(hiddenUnitPreSigmoid);
            float[] outputPreSoftmax = new float[3];
            // Normally a matrix multiplication here, but only one hidden unit in this trivial example
            for (int i = 0; i < 3; i++) {
                outputPreSoftmax[i] = hiddenUnitPostSigmoid * l2WeightsFloat[i] + l2BiasFloatArray[i];
            }
            float[] outputPostSoftmax = softmax(outputPreSoftmax);
            // Do backward pass:
            // out-labels
            float[] deltaOut = vectorDifference(outputPostSoftmax, yFloat);
            // deltaHidden = sigmaPrime(hiddenUnitZ) * sum_k (w_jk * \delta_k); here, only one j
            float deltaHidden = 0.0f;
            for (int i = 0; i < 3; i++) deltaHidden += l2WeightsFloat[i] * deltaOut[i];
            deltaHidden *= derivOfSigmoid(hiddenUnitPreSigmoid);
            // Calculate weight/bias updates:
            // dL/dW = delta * (activation of prev. layer)
            // dL/db = delta
            float[] dLdwOut = new float[3];
            for (int i = 0; i < dLdwOut.length; i++) dLdwOut[i] = deltaOut[i] * hiddenUnitPostSigmoid;
            float[] dLdwHidden = new float[4];
            for (int i = 0; i < dLdwHidden.length; i++) dLdwHidden[i] = deltaHidden * xFloat[i];
            float[] dLdbOut = deltaOut;
            float dLdbHidden = deltaHidden;
            if (printCalculations) {
                log.debug("deltaOut = {}", Arrays.toString(deltaOut));
                log.debug("deltaHidden = {}", deltaHidden);
                log.debug("dLdwOut = {}", Arrays.toString(dLdwOut));
                log.debug("dLdbOut = {}", Arrays.toString(dLdbOut));
                log.debug("dLdwHidden = {}", Arrays.toString(dLdwHidden));
                log.debug("dLdbHidden = {}", dLdbHidden);
            }
            // Calculate new parameters:
            // w_i = w_i - (learningRate)/(batchSize) * sum_j (dL_j/dw_i)
            // b_i = b_i - (learningRate)/(batchSize) * sum_j (dL_j/db_i)
            // Which for batch size of one (here) is simply:
            // w_i = w_i - learningRate * dL/dW
            // b_i = b_i - learningRate * dL/db
            float[] expectedL1WeightsAfter = new float[4];
            float[] expectedL2WeightsAfter = new float[3];
            float expectedL1BiasAfter = l1BiasFloat - 0.1f * dLdbHidden;
            float[] expectedL2BiasAfter = new float[3];
            for (int i = 0; i < 4; i++) expectedL1WeightsAfter[i] = l1WeightsFloat[i] - 0.1f * dLdwHidden[i];
            for (int i = 0; i < 3; i++) expectedL2WeightsAfter[i] = l2WeightsFloat[i] - 0.1f * dLdwOut[i];
            for (int i = 0; i < 3; i++) expectedL2BiasAfter[i] = l2BiasFloatArray[i] - 0.1f * dLdbOut[i];
            // Finally, do back-prop on network, and compare parameters vs. expected parameters
            network.fit(data);
            /*  INDArray l1WeightsAfter = layers[0].getParam(DefaultParamInitializer.WEIGHT_KEY).dup();	//Hidden layer
            INDArray l2WeightsAfter = layers[1].getParam(DefaultParamInitializer.WEIGHT_KEY).dup();	//Output layer
            INDArray l1BiasAfter = layers[0].getParam(DefaultParamInitializer.BIAS_KEY).dup();
            INDArray l2BiasAfter = layers[1].getParam(DefaultParamInitializer.BIAS_KEY).dup();
            float[] l1WeightsFloatAfter = asFloat(l1WeightsAfter);
            float[] l2WeightsFloatAfter = asFloat(l2WeightsAfter);
            float l1BiasFloatAfter = l1BiasAfter.getFloat(0);
            float[] l2BiasFloatAfter = asFloat(l2BiasAfter);
            
            if( printCalculations) {
                log.debug("Expected L1 weights = {}", Arrays.toString(expectedL1WeightsAfter));
                log.debug("Actual L1 weights = {}", Arrays.toString(asFloat(l1WeightsAfter)));
                log.debug("Expected L2 weights = {}", Arrays.toString(expectedL2WeightsAfter));
                log.debug("Actual L2 weights = {}", Arrays.toString(asFloat(l2WeightsAfter)));
                log.debug("Expected L1 bias = {}", expectedL1BiasAfter);
                log.debug("Actual L1 bias = {}", Arrays.toString(asFloat(l1BiasAfter)));
                log.debug("Expected L2 bias = {}", Arrays.toString(expectedL2BiasAfter));
                log.debug("Actual L2 bias = {}", Arrays.toString(asFloat(l2BiasAfter)));
            }
            
            
            float eps = 1e-4f;
            assertArrayEquals(l1WeightsFloatAfter,expectedL1WeightsAfter,eps);
            assertArrayEquals(l2WeightsFloatAfter,expectedL2WeightsAfter,eps);
            assertEquals(l1BiasFloatAfter,expectedL1BiasAfter,eps);
            assertArrayEquals(l2BiasFloatAfter,expectedL2BiasAfter,eps);
            */
            // System.out.println("\n\n--------------");
        }
    }

    @Test
    @DisplayName("Test MLP Gradient Calculation")
    void testMLPGradientCalculation() {
        testIrisMiniBatchGradients(1, new int[] { 1 }, Activation.SIGMOID);
        testIrisMiniBatchGradients(1, new int[] { 5 }, Activation.SIGMOID);
        testIrisMiniBatchGradients(12, new int[] { 15, 25, 10 }, Activation.SIGMOID);
        testIrisMiniBatchGradients(50, new int[] { 10, 50, 200, 50, 10 }, Activation.TANH);
        testIrisMiniBatchGradients(150, new int[] { 30, 50, 20 }, Activation.TANH);
    }

    private static void testIrisMiniBatchGradients(int miniBatchSize, int[] hiddenLayerSizes, Activation activationFunction) {
        int totalExamples = 10 * miniBatchSize;
        if (totalExamples > 150) {
            totalExamples = miniBatchSize * (150 / miniBatchSize);
        }
        if (miniBatchSize > 150) {
            fail();
        }
        DataSetIterator iris = new IrisDataSetIterator(miniBatchSize, totalExamples);
        MultiLayerNetwork network = new MultiLayerNetwork(getIrisMLPSimpleConfig(hiddenLayerSizes, Activation.SIGMOID));
        network.init();
        Layer[] layers = network.getLayers();
        int nLayers = layers.length;
        while (iris.hasNext()) {
            DataSet data = iris.next();
            INDArray x = data.getFeatures();
            INDArray y = data.getLabels();
            // Do forward pass:
            INDArray[] layerWeights = new INDArray[nLayers];
            INDArray[] layerBiases = new INDArray[nLayers];
            for (int i = 0; i < nLayers; i++) {
                layerWeights[i] = layers[i].getParam(DefaultParamInitializer.WEIGHT_KEY).dup();
                layerBiases[i] = layers[i].getParam(DefaultParamInitializer.BIAS_KEY).dup();
            }
            INDArray[] layerZs = new INDArray[nLayers];
            INDArray[] layerActivations = new INDArray[nLayers];
            for (int i = 0; i < nLayers; i++) {
                INDArray layerInput = (i == 0 ? x : layerActivations[i - 1]);
                layerZs[i] = layerInput.castTo(layerWeights[i].dataType()).mmul(layerWeights[i]).addiRowVector(layerBiases[i]);
                layerActivations[i] = (i == nLayers - 1 ? doSoftmax(layerZs[i].dup()) : doSigmoid(layerZs[i].dup()));
            }
            // Do backward pass:
            INDArray[] deltas = new INDArray[nLayers];
            // Out - labels; shape=[miniBatchSize,nOut];
            deltas[nLayers - 1] = layerActivations[nLayers - 1].sub(y.castTo(layerActivations[nLayers - 1].dataType()));
            assertArrayEquals(deltas[nLayers - 1].shape(), new long[] { miniBatchSize, 3 });
            for (int i = nLayers - 2; i >= 0; i--) {
                INDArray sigmaPrimeOfZ;
                sigmaPrimeOfZ = doSigmoidDerivative(layerZs[i]);
                INDArray epsilon = layerWeights[i + 1].mmul(deltas[i + 1].transpose()).transpose();
                deltas[i] = epsilon.mul(sigmaPrimeOfZ);
                assertArrayEquals(deltas[i].shape(), new long[] { miniBatchSize, hiddenLayerSizes[i] });
            }
            INDArray[] dLdw = new INDArray[nLayers];
            INDArray[] dLdb = new INDArray[nLayers];
            for (int i = 0; i < nLayers; i++) {
                INDArray prevActivations = (i == 0 ? x : layerActivations[i - 1]);
                // Raw gradients, so not yet divided by mini-batch size (division is done in BaseUpdater)
                // Shape: [nIn, nOut]
                dLdw[i] = deltas[i].transpose().castTo(prevActivations.dataType()).mmul(prevActivations).transpose();
                // Shape: [1,nOut]
                dLdb[i] = deltas[i].sum(true, 0);
                int nIn = (i == 0 ? 4 : hiddenLayerSizes[i - 1]);
                int nOut = (i < nLayers - 1 ? hiddenLayerSizes[i] : 3);
                assertArrayEquals(dLdw[i].shape(), new long[] { nIn, nOut });
                assertArrayEquals(dLdb[i].shape(), new long[] { 1, nOut });
            }
            // Calculate and get gradient, compare to expected
            network.setInput(x);
            network.setLabels(y);
            network.computeGradientAndScore();
            Gradient gradient = network.gradientAndScore().getFirst();
            float eps = 1e-4f;
            for (int i = 0; i < hiddenLayerSizes.length; i++) {
                String wKey = i + "_" + DefaultParamInitializer.WEIGHT_KEY;
                String bKey = i + "_" + DefaultParamInitializer.BIAS_KEY;
                INDArray wGrad = gradient.getGradientFor(wKey);
                INDArray bGrad = gradient.getGradientFor(bKey);
                float[] wGradf = asFloat(wGrad);
                float[] bGradf = asFloat(bGrad);
                float[] expWGradf = asFloat(dLdw[i]);
                float[] expBGradf = asFloat(dLdb[i]);
                assertArrayEquals(wGradf, expWGradf, eps);
                assertArrayEquals(bGradf, expBGradf, eps);
            }
        }
    }

    /**
     * Very simple back-prop config set up for Iris.
     * Learning Rate = 0.1
     * No regularization, no Adagrad, no momentum etc. One iteration.
     */
    private static MultiLayerConfiguration getIrisMLPSimpleConfig(int[] hiddenLayerSizes, Activation activationFunction) {
        ListBuilder lb = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1)).seed(12345L).trainingWorkspaceMode(org.deeplearning4j.nn.conf.WorkspaceMode.NONE).list();
        for (int i = 0; i < hiddenLayerSizes.length; i++) {
            int nIn = (i == 0 ? 4 : hiddenLayerSizes[i - 1]);
            lb.layer(i, new DenseLayer.Builder().nIn(nIn).nOut(hiddenLayerSizes[i]).weightInit(WeightInit.XAVIER).activation(activationFunction).build());
        }
        lb.layer(hiddenLayerSizes.length, new OutputLayer.Builder(LossFunction.MCXENT).nIn(hiddenLayerSizes[hiddenLayerSizes.length - 1]).nOut(3).weightInit(WeightInit.XAVIER).activation(activationFunction.equals(Activation.IDENTITY) ? Activation.IDENTITY : Activation.SOFTMAX).build());
        return lb.build();
    }

    public static float[] asFloat(INDArray arr) {
        long len = arr.length();
        if (len > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();
        float[] f = new float[(int) len];
        NdIndexIterator iterator = new NdIndexIterator('c', arr.shape());
        for (int i = 0; i < len; i++) {
            f[i] = arr.getFloat(iterator.next());
        }
        return f;
    }

    public static float dotProduct(float[] x, float[] y) {
        float sum = 0.0f;
        for (int i = 0; i < x.length; i++) sum += x[i] * y[i];
        return sum;
    }

    public static float sigmoid(float in) {
        return (float) (1.0 / (1.0 + Math.exp(-in)));
    }

    public static float[] sigmoid(float[] in) {
        float[] out = new float[in.length];
        for (int i = 0; i < in.length; i++) {
            out[i] = sigmoid(in[i]);
        }
        return out;
    }

    public static float derivOfSigmoid(float in) {
        // float v = (float)( Math.exp(in) / Math.pow(1+Math.exp(in),2.0) );
        float v = in * (1 - in);
        return v;
    }

    public static float[] derivOfSigmoid(float[] in) {
        float[] out = new float[in.length];
        for (int i = 0; i < in.length; i++) {
            out[i] = derivOfSigmoid(in[i]);
        }
        return out;
    }

    public static float[] softmax(float[] in) {
        float[] out = new float[in.length];
        float sumExp = 0.0f;
        for (int i = 0; i < in.length; i++) {
            sumExp += Math.exp(in[i]);
        }
        for (int i = 0; i < in.length; i++) {
            out[i] = (float) Math.exp(in[i]) / sumExp;
        }
        return out;
    }

    public static float[] vectorDifference(float[] x, float[] y) {
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i] - y[i];
        }
        return out;
    }

    public static INDArray doSoftmax(INDArray input) {
        return Transforms.softmax(input, true);
    }

    public static INDArray doSigmoid(INDArray input) {
        return Transforms.sigmoid(input, true);
    }

    public static INDArray doSigmoidDerivative(INDArray input) {
        return Nd4j.getExecutioner().exec(new SigmoidDerivative(input.dup()));
    }

    @Test
    @DisplayName("Debug Forward Pass Comparison")
    void debugForwardPassComparison() {
        // Reproduce test case 2: (1, [5])
        int miniBatchSize = 1;
        int[] hiddenLayerSizes = new int[] { 5 };
        DataSetIterator iris = new IrisDataSetIterator(miniBatchSize, miniBatchSize);
        MultiLayerNetwork network = new MultiLayerNetwork(getIrisMLPSimpleConfig(hiddenLayerSizes, Activation.SIGMOID));
        network.init();
        Layer[] layers = network.getLayers();
        int nLayers = layers.length;

        DataSet data = iris.next();
        INDArray x = data.getFeatures();
        INDArray y = data.getLabels();

        // Get weights
        INDArray W0 = layers[0].getParam(DefaultParamInitializer.WEIGHT_KEY).dup();
        INDArray b0 = layers[0].getParam(DefaultParamInitializer.BIAS_KEY).dup();
        INDArray W1 = layers[1].getParam(DefaultParamInitializer.WEIGHT_KEY).dup();
        INDArray b1 = layers[1].getParam(DefaultParamInitializer.BIAS_KEY).dup();

        log.debug("x shape={} ordering={} strides={}", java.util.Arrays.toString(x.shape()), x.ordering(), java.util.Arrays.toString(x.stride()));
        log.debug("W0 shape={} ordering={} strides={}", java.util.Arrays.toString(W0.shape()), W0.ordering(), java.util.Arrays.toString(W0.stride()));
        log.debug("W0_orig shape={} ordering={} strides={} isView={}", java.util.Arrays.toString(layers[0].getParam(DefaultParamInitializer.WEIGHT_KEY).shape()), layers[0].getParam(DefaultParamInitializer.WEIGHT_KEY).ordering(), java.util.Arrays.toString(layers[0].getParam(DefaultParamInitializer.WEIGHT_KEY).stride()), layers[0].getParam(DefaultParamInitializer.WEIGHT_KEY).isView());
        log.debug("W0 row0={}", W0.getRow(0));
        log.debug("W0_orig row0={}", layers[0].getParam(DefaultParamInitializer.WEIGHT_KEY).getRow(0));
        log.debug("W1 shape={} ordering={} strides={}", java.util.Arrays.toString(W1.shape()), W1.ordering(), java.util.Arrays.toString(W1.stride()));

        // Manual forward pass - use 'f' ordering to match network
        INDArray W0f = W0.dup('f');
        INDArray W1f = W1.dup('f');
        INDArray b0f = b0.dup('f');
        INDArray b1f = b1.dup('f');
        INDArray xCast = x.castTo(W0.dataType());
        log.debug("xCast ordering={} strides={}", xCast.ordering(), java.util.Arrays.toString(xCast.stride()));
        // Compare: mmul with 'c' vs 'f' weights
        INDArray z0c = xCast.mmul(W0);
        INDArray z0f = xCast.mmul(W0f);
        log.debug("z0c (c-order W)={}", z0c);
        log.debug("z0f (f-order W)={}", z0f);
        log.debug("z0 diff={}", z0c.sub(z0f).norm2Number().doubleValue());
        // Check data contents of W0 (c) vs W0f (f)
        log.debug("W0c row0={} row1={}", W0.getRow(0), W0.getRow(1));
        log.debug("W0f row0={} row1={}", W0f.getRow(0), W0f.getRow(1));
        log.debug("W0c col0={} col1={}", W0.getColumn(0), W0.getColumn(1));
        log.debug("W0f col0={} col1={}", W0f.getColumn(0), W0f.getColumn(1));
        // Try Nd4j.gemm directly
        INDArray gemmResultC = Nd4j.createUninitialized(xCast.dataType(), new long[]{1, 5}, 'f');
        Nd4j.gemm(xCast, W0, gemmResultC, false, false, 1.0, 0.0);
        log.debug("gemm(x,W0c)={}", gemmResultC);
        INDArray gemmResultF = Nd4j.createUninitialized(xCast.dataType(), new long[]{1, 5}, 'f');
        Nd4j.gemm(xCast, W0f, gemmResultF, false, false, 1.0, 0.0);
        log.debug("gemm(x,W0f)={}", gemmResultF);
        INDArray z0 = z0f.addiRowVector(b0f);
        log.debug("z0 ordering={} vals={}", z0.ordering(), z0);
        INDArray a0 = doSigmoid(z0.dup());
        log.debug("a0 ordering={} vals={}", a0.ordering(), a0);

        INDArray z1 = a0.castTo(W1.dataType()).mmul(W1).addiRowVector(b1);
        log.debug("z1 ordering={} vals={}", z1.ordering(), z1);
        INDArray a1_manual = doSoftmax(z1.dup());
        log.debug("a1_manual ordering={} vals={}", a1_manual.ordering(), a1_manual);

        // Network forward pass
        network.setInput(x);
        java.util.List<INDArray> netActs = network.feedForward(false);
        INDArray netA0 = netActs.get(1);
        INDArray netA1 = netActs.get(2);
        log.debug("netA0 ordering={} vals={}", netA0.ordering(), netA0);
        log.debug("netA1 ordering={} vals={}", netA1.ordering(), netA1);

        double a0Diff = a0.sub(netA0).norm2Number().doubleValue();
        double a1Diff = a1_manual.sub(netA1).norm2Number().doubleValue();
        log.debug("a0 diff={}", a0Diff);
        log.debug("a1 diff={}", a1Diff);

        // Also compare delta
        INDArray manualDelta = a1_manual.sub(y.castTo(a1_manual.dataType()));
        INDArray netDelta = netA1.sub(y.castTo(netA1.dataType()));
        log.debug("manual delta={}", manualDelta);
        log.debug("net delta={}", netDelta);
        double deltaDiff = manualDelta.sub(netDelta).norm2Number().doubleValue();
        log.debug("delta diff={}", deltaDiff);
    }
}
