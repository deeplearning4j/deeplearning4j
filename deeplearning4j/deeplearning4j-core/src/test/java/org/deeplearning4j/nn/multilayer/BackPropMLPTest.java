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

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SigmoidDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.TanhDerivative;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.fail;

public class BackPropMLPTest extends BaseDL4JTest {

    @Test
    public void testMLPTrivial() {
        //Simplest possible case: 1 hidden layer, 1 hidden neuron, batch size of 1.
        MultiLayerNetwork network = new MultiLayerNetwork(getIrisMLPSimpleConfig(new int[] {1}, Activation.SIGMOID));
        network.setListeners(new ScoreIterationListener(1));
        network.init();

        DataSetIterator iter = new IrisDataSetIterator(1, 10);

        while (iter.hasNext())
            network.fit(iter.next());
    }

    @Test
    public void testMLP() {
        //Simple mini-batch test with multiple hidden layers
        MultiLayerConfiguration conf = getIrisMLPSimpleConfig(new int[] {5, 4, 3}, Activation.SIGMOID);
        System.out.println(conf);
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        DataSetIterator iter = new IrisDataSetIterator(10, 100);

        while (iter.hasNext()) {
            network.fit(iter.next());
        }
    }

    @Test
    public void testMLP2() {
        //Simple mini-batch test with multiple hidden layers
        MultiLayerConfiguration conf = getIrisMLPSimpleConfig(new int[] {5, 15, 3}, Activation.TANH);
        System.out.println(conf);
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        DataSetIterator iter = new IrisDataSetIterator(12, 120);

        while (iter.hasNext()) {
            network.fit(iter.next());
        }
    }

    @Test
    public void testSingleExampleWeightUpdates() {
        //Simplest possible case: 1 hidden layer, 1 hidden neuron, batch size of 1.
        //Manually calculate weight updates (entirely outside of DL4J and ND4J)
        // and compare expected and actual weights after backprop

        DataSetIterator iris = new IrisDataSetIterator(1, 10);

        MultiLayerNetwork network = new MultiLayerNetwork(getIrisMLPSimpleConfig(new int[] {1}, Activation.SIGMOID));
        network.init();

        Layer[] layers = network.getLayers();

        final boolean printCalculations = true;

        while (iris.hasNext()) {
            DataSet data = iris.next();
            INDArray x = data.getFeatures();
            INDArray y = data.getLabels();
            float[] xFloat = asFloat(x);
            float[] yFloat = asFloat(y);

            //Do forward pass:
            INDArray l1Weights = layers[0].getParam(DefaultParamInitializer.WEIGHT_KEY).dup(); //Hidden layer
            INDArray l2Weights = layers[1].getParam(DefaultParamInitializer.WEIGHT_KEY).dup(); //Output layer
            INDArray l1Bias = layers[0].getParam(DefaultParamInitializer.BIAS_KEY).dup();
            INDArray l2Bias = layers[1].getParam(DefaultParamInitializer.BIAS_KEY).dup();
            float[] l1WeightsFloat = asFloat(l1Weights);
            float[] l2WeightsFloat = asFloat(l2Weights);
            float l1BiasFloat = l1Bias.getFloat(0);
            float[] l2BiasFloatArray = asFloat(l2Bias);

            float hiddenUnitPreSigmoid = dotProduct(l1WeightsFloat, xFloat) + l1BiasFloat; //z=w*x+b
            float hiddenUnitPostSigmoid = sigmoid(hiddenUnitPreSigmoid); //a=sigma(z)

            float[] outputPreSoftmax = new float[3];
            //Normally a matrix multiplication here, but only one hidden unit in this trivial example
            for (int i = 0; i < 3; i++) {
                outputPreSoftmax[i] = hiddenUnitPostSigmoid * l2WeightsFloat[i] + l2BiasFloatArray[i];
            }
            float[] outputPostSoftmax = softmax(outputPreSoftmax);

            //Do backward pass:
            float[] deltaOut = vectorDifference(outputPostSoftmax, yFloat); //out-labels
            //deltaHidden = sigmaPrime(hiddenUnitZ) * sum_k (w_jk * \delta_k); here, only one j
            float deltaHidden = 0.0f;
            for (int i = 0; i < 3; i++)
                deltaHidden += l2WeightsFloat[i] * deltaOut[i];
            deltaHidden *= derivOfSigmoid(hiddenUnitPreSigmoid);

            //Calculate weight/bias updates:
            //dL/dW = delta * (activation of prev. layer)
            //dL/db = delta
            float[] dLdwOut = new float[3];
            for (int i = 0; i < dLdwOut.length; i++)
                dLdwOut[i] = deltaOut[i] * hiddenUnitPostSigmoid;
            float[] dLdwHidden = new float[4];
            for (int i = 0; i < dLdwHidden.length; i++)
                dLdwHidden[i] = deltaHidden * xFloat[i];
            float[] dLdbOut = deltaOut;
            float dLdbHidden = deltaHidden;

            if (printCalculations) {
                System.out.println("deltaOut = " + Arrays.toString(deltaOut));
                System.out.println("deltaHidden = " + deltaHidden);
                System.out.println("dLdwOut = " + Arrays.toString(dLdwOut));
                System.out.println("dLdbOut = " + Arrays.toString(dLdbOut));
                System.out.println("dLdwHidden = " + Arrays.toString(dLdwHidden));
                System.out.println("dLdbHidden = " + dLdbHidden);
            }


            //Calculate new parameters:
            //w_i = w_i - (learningRate)/(batchSize) * sum_j (dL_j/dw_i)
            //b_i = b_i - (learningRate)/(batchSize) * sum_j (dL_j/db_i)
            //Which for batch size of one (here) is simply:
            //w_i = w_i - learningRate * dL/dW
            //b_i = b_i - learningRate * dL/db
            float[] expectedL1WeightsAfter = new float[4];
            float[] expectedL2WeightsAfter = new float[3];
            float expectedL1BiasAfter = l1BiasFloat - 0.1f * dLdbHidden;
            float[] expectedL2BiasAfter = new float[3];

            for (int i = 0; i < 4; i++)
                expectedL1WeightsAfter[i] = l1WeightsFloat[i] - 0.1f * dLdwHidden[i];
            for (int i = 0; i < 3; i++)
                expectedL2WeightsAfter[i] = l2WeightsFloat[i] - 0.1f * dLdwOut[i];
            for (int i = 0; i < 3; i++)
                expectedL2BiasAfter[i] = l2BiasFloatArray[i] - 0.1f * dLdbOut[i];


            //Finally, do back-prop on network, and compare parameters vs. expected parameters
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
                System.out.println("Expected L1 weights = " + Arrays.toString(expectedL1WeightsAfter));
                System.out.println("Actual L1 weights = " + Arrays.toString(asFloat(l1WeightsAfter)));
                System.out.println("Expected L2 weights = " + Arrays.toString(expectedL2WeightsAfter));
                System.out.println("Actual L2 weights = " + Arrays.toString(asFloat(l2WeightsAfter)));
                System.out.println("Expected L1 bias = " + expectedL1BiasAfter);
                System.out.println("Actual L1 bias = " + Arrays.toString(asFloat(l1BiasAfter)));
                System.out.println("Expected L2 bias = " + Arrays.toString(expectedL2BiasAfter));
                System.out.println("Actual L2 bias = " + Arrays.toString(asFloat(l2BiasAfter)));
            }
            
            
            float eps = 1e-4f;
            assertArrayEquals(l1WeightsFloatAfter,expectedL1WeightsAfter,eps);
            assertArrayEquals(l2WeightsFloatAfter,expectedL2WeightsAfter,eps);
            assertEquals(l1BiasFloatAfter,expectedL1BiasAfter,eps);
            assertArrayEquals(l2BiasFloatAfter,expectedL2BiasAfter,eps);
            */
            System.out.println("\n\n--------------");
        }
    }


    @Test
    public void testMLPGradientCalculation() {
        testIrisMiniBatchGradients(1, new int[] {1}, Activation.SIGMOID);
        testIrisMiniBatchGradients(1, new int[] {5}, Activation.SIGMOID);
        testIrisMiniBatchGradients(12, new int[] {15, 25, 10}, Activation.SIGMOID);
        testIrisMiniBatchGradients(50, new int[] {10, 50, 200, 50, 10}, Activation.TANH);
        testIrisMiniBatchGradients(150, new int[] {30, 50, 20}, Activation.TANH);
    }

    private static void testIrisMiniBatchGradients(int miniBatchSize, int[] hiddenLayerSizes,
                    Activation activationFunction) {
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

            //Do forward pass:
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
                layerZs[i] = layerInput.mmul(layerWeights[i]).addiRowVector(layerBiases[i]);
                layerActivations[i] = (i == nLayers - 1 ? doSoftmax(layerZs[i].dup()) : doSigmoid(layerZs[i].dup()));
            }

            //Do backward pass:
            INDArray[] deltas = new INDArray[nLayers];
            deltas[nLayers - 1] = layerActivations[nLayers - 1].sub(y); //Out - labels; shape=[miniBatchSize,nOut];
            assertArrayEquals(deltas[nLayers - 1].shape(), new long[] {miniBatchSize, 3});
            for (int i = nLayers - 2; i >= 0; i--) {
                INDArray sigmaPrimeOfZ;
                sigmaPrimeOfZ = doSigmoidDerivative(layerZs[i]);
                INDArray epsilon = layerWeights[i + 1].mmul(deltas[i + 1].transpose()).transpose();
                deltas[i] = epsilon.mul(sigmaPrimeOfZ);
                assertArrayEquals(deltas[i].shape(), new long[] {miniBatchSize, hiddenLayerSizes[i]});
            }

            INDArray[] dLdw = new INDArray[nLayers];
            INDArray[] dLdb = new INDArray[nLayers];
            for (int i = 0; i < nLayers; i++) {
                INDArray prevActivations = (i == 0 ? x : layerActivations[i - 1]);
                //Raw gradients, so not yet divided by mini-batch size (division is done in BaseUpdater)
                dLdw[i] = deltas[i].transpose().mmul(prevActivations).transpose(); //Shape: [nIn, nOut]
                dLdb[i] = deltas[i].sum(0); //Shape: [1,nOut]

                int nIn = (i == 0 ? 4 : hiddenLayerSizes[i - 1]);
                int nOut = (i < nLayers - 1 ? hiddenLayerSizes[i] : 3);
                assertArrayEquals(dLdw[i].shape(), new long[] {nIn, nOut});
                assertArrayEquals(dLdb[i].shape(), new long[] {1, nOut});
            }


            //Calculate and get gradient, compare to expected
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


    /** Very simple back-prop config set up for Iris.
     * Learning Rate = 0.1
     * No regularization, no Adagrad, no momentum etc. One iteration.
     */
    private static MultiLayerConfiguration getIrisMLPSimpleConfig(int[] hiddenLayerSizes,
                    Activation activationFunction) {
        NeuralNetConfiguration.ListBuilder lb = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1))
                    .seed(12345L).list();

        for (int i = 0; i < hiddenLayerSizes.length; i++) {
            int nIn = (i == 0 ? 4 : hiddenLayerSizes[i - 1]);
            lb.layer(i, new DenseLayer.Builder().nIn(nIn).nOut(hiddenLayerSizes[i]).weightInit(WeightInit.XAVIER)
                            .activation(activationFunction).build());
        }

        lb.layer(hiddenLayerSizes.length,
                        new OutputLayer.Builder(LossFunction.MCXENT).nIn(hiddenLayerSizes[hiddenLayerSizes.length - 1])
                                        .nOut(3).weightInit(WeightInit.XAVIER)
                                        .activation(activationFunction.equals(Activation.IDENTITY) ? Activation.IDENTITY
                                                        : Activation.SOFTMAX)
                                        .build());
        lb.pretrain(false).backprop(true);

        return lb.build();
    }

    public static float[] asFloat(INDArray arr) {
        long len = arr.length();
        // FIXME: int cast
        float[] f = new float[(int) len];
        NdIndexIterator iterator = new NdIndexIterator('c', arr.shape());
        for (int i = 0; i < len; i++) {
            f[i] = arr.getFloat(iterator.next());
        }
        return f;
    }

    public static float dotProduct(float[] x, float[] y) {
        float sum = 0.0f;
        for (int i = 0; i < x.length; i++)
            sum += x[i] * y[i];
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
        //		float v = (float)( Math.exp(in) / Math.pow(1+Math.exp(in),2.0) );
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

    public static INDArray doTanh(INDArray input) {
        return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", input.dup()));
    }

    public static INDArray doTanhDerivative(INDArray input) {
        return Nd4j.getExecutioner()
                        .execAndReturn(new TanhDerivative(input.dup()));
    }

    public static INDArray doSoftmax(INDArray input) {
        return Transforms.softmax(input, true);
    }

    public static INDArray doSigmoid(INDArray input) {
        return Transforms.sigmoid(input, true);
    }

    public static INDArray doSigmoidDerivative(INDArray input) {
        return Nd4j.getExecutioner().execAndReturn(new SigmoidDerivative(input.dup()));
    }

}
