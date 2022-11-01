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
package org.eclipse.deeplearning4j.dl4jcore.gradientcheck;

import org.deeplearning4j.BaseDL4JTest;
import org.eclipse.deeplearning4j.dl4jcore.TestUtils;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.DisplayName;

/**
 */
@DisplayName("Bn Gradient Check Test")
@NativeTag
@Tag(TagNames.NDARRAY_ETL)
@Tag(TagNames.TRAINING)
@Tag(TagNames.DL4J_OLD_API)
class BNGradientCheckTest extends BaseDL4JTest {

    static {
        Nd4j.setDataType(DataType.DOUBLE);
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 90000L;
    }

    @Test
    @DisplayName("Test Gradient 2 d Simple")
    void testGradient2dSimple() {
        DataNormalization scaler = new NormalizerMinMaxScaler();
        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        scaler.fit(iter);
        iter.setPreProcessor(scaler);
        DataSet ds = iter.next();
        INDArray input = ds.getFeatures();
        INDArray labels = ds.getLabels();
        for (boolean useLogStd : new boolean[] { true, false }) {
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().updater(new NoOp()).dataType(DataType.DOUBLE).seed(12345L).dist(new NormalDistribution(0, 1)).list().layer(0, new DenseLayer.Builder().nIn(4).nOut(3).activation(Activation.IDENTITY).build()).layer(1, new BatchNormalization.Builder().useLogStd(useLogStd).nOut(3).build()).layer(2, new ActivationLayer.Builder().activation(Activation.TANH).build()).layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(3).nOut(3).build());
            MultiLayerNetwork mln = new MultiLayerNetwork(builder.build());
            mln.init();
            // for (int j = 0; j < mln.getnLayers(); j++)
            // System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
            // Mean and variance vars are not gradient checkable; mean/variance "gradient" is used to implement running mean/variance calc
            // i.e., runningMean = decay * runningMean + (1-decay) * batchMean
            // However, numerical gradient will be 0 as forward pass doesn't depend on this "parameter"
            Set<String> excludeParams = new HashSet<>(Arrays.asList("1_mean", "1_var", "1_log10stdev"));
            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.MLNConfig().net(mln).input(input).labels(labels).excludeParams(excludeParams));
            assertTrue(gradOK);
            TestUtils.testModelSerialization(mln);
        }
    }

    @Test
    @DisplayName("Test Gradient Cnn Simple")
    void testGradientCnnSimple() {
        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int depth = 1;
        int hw = 4;
        int nOut = 4;
        INDArray input = Nd4j.rand(new int[] { minibatch, depth, hw, hw });
        INDArray labels = Nd4j.zeros(minibatch, nOut);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            labels.putScalar(i, r.nextInt(nOut), 1.0);
        }
        for (boolean useLogStd : new boolean[] { true, false }) {
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration
                    .Builder().dataType(DataType.DOUBLE)
                    .trainingWorkspaceMode(WorkspaceMode.NONE)
                    .updater(new NoOp()).seed(12345L)
                    .dist(new NormalDistribution(0, 2))
                    .list().layer(0, new ConvolutionLayer.Builder()
                            .kernelSize(2, 2).stride(1, 1)
                            .nIn(depth).nOut(2).activation(Activation.IDENTITY)
                            .build())
                    .layer(1, new BatchNormalization.Builder().useLogStd(useLogStd).build())
                    .layer(2, new ActivationLayer.Builder().activation(Activation.TANH).build())
                    .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                            .activation(Activation.SOFTMAX).nOut(nOut).build())
                    .setInputType(InputType.convolutional(hw, hw, depth));
            MultiLayerNetwork mln = new MultiLayerNetwork(builder.build());
            mln.init();
            // for (int j = 0; j < mln.getnLayers(); j++)
            // System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
            // Mean and variance vars are not gradient checkable; mean/variance "gradient" is used to implement running mean/variance calc
            // i.e., runningMean = decay * runningMean + (1-decay) * batchMean
            // However, numerical gradient will be 0 as forward pass doesn't depend on this "parameter"
            Set<String> excludeParams = new HashSet<>(Arrays.asList("1_mean", "1_var", "1_log10stdev"));
            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.MLNConfig().net(mln).input(input).labels(labels).excludeParams(excludeParams));
            assertTrue(gradOK);
            TestUtils.testModelSerialization(mln);
        }
    }





    @Test
    @DisplayName("Test Gradient 2 d Fixed Gamma Beta")
    void testGradient2dFixedGammaBeta() {
        DataNormalization scaler = new NormalizerMinMaxScaler();
        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        scaler.fit(iter);
        iter.setPreProcessor(scaler);
        DataSet ds = iter.next();
        INDArray input = ds.getFeatures();
        INDArray labels = ds.getLabels();
        for (boolean useLogStd : new boolean[] { true, false }) {
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().updater(new NoOp()).dataType(DataType.DOUBLE).seed(12345L).dist(new NormalDistribution(0, 1)).list().layer(0, new DenseLayer.Builder().nIn(4).nOut(3).activation(Activation.IDENTITY).build()).layer(1, new BatchNormalization.Builder().useLogStd(useLogStd).lockGammaBeta(true).gamma(2.0).beta(0.5).nOut(3).build()).layer(2, new ActivationLayer.Builder().activation(Activation.TANH).build()).layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(3).nOut(3).build());
            MultiLayerNetwork mln = new MultiLayerNetwork(builder.build());
            mln.init();
            // for (int j = 0; j < mln.getnLayers(); j++)
            // System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
            // Mean and variance vars are not gradient checkable; mean/variance "gradient" is used to implement running mean/variance calc
            // i.e., runningMean = decay * runningMean + (1-decay) * batchMean
            // However, numerical gradient will be 0 as forward pass doesn't depend on this "parameter"
            Set<String> excludeParams = new HashSet<>(Arrays.asList("1_mean", "1_var", "1_log10stdev"));
            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.MLNConfig().net(mln).input(input).labels(labels).excludeParams(excludeParams));
            assertTrue(gradOK);
            TestUtils.testModelSerialization(mln);
        }
    }

    @Test
    @DisplayName("Test Gradient Cnn Fixed Gamma Beta")
    void testGradientCnnFixedGammaBeta() {
        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int depth = 1;
        int hw = 4;
        int nOut = 4;
        INDArray input = Nd4j.rand(new int[] { minibatch, depth, hw, hw });
        INDArray labels = Nd4j.zeros(minibatch, nOut);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            labels.putScalar(i, r.nextInt(nOut), 1.0);
        }
        for (boolean useLogStd : new boolean[] { true, false }) {
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().updater(new NoOp()).dataType(DataType.DOUBLE).seed(12345L).dist(new NormalDistribution(0, 2)).list().layer(0, new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).nIn(depth).nOut(2).activation(Activation.IDENTITY).build()).layer(1, new BatchNormalization.Builder().useLogStd(useLogStd).lockGammaBeta(true).gamma(2.0).beta(0.5).build()).layer(2, new ActivationLayer.Builder().activation(Activation.TANH).build()).layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nOut(nOut).build()).setInputType(InputType.convolutional(hw, hw, depth));
            MultiLayerNetwork mln = new MultiLayerNetwork(builder.build());
            mln.init();
            // for (int j = 0; j < mln.getnLayers(); j++)
            // System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
            // Mean and variance vars are not gradient checkable; mean/variance "gradient" is used to implement running mean/variance calc
            // i.e., runningMean = decay * runningMean + (1-decay) * batchMean
            // However, numerical gradient will be 0 as forward pass doesn't depend on this "parameter"
            Set<String> excludeParams = new HashSet<>(Arrays.asList("1_mean", "1_var", "1_log10stdev"));
            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.MLNConfig().net(mln).input(input).labels(labels).excludeParams(excludeParams));
            assertTrue(gradOK);
            TestUtils.testModelSerialization(mln);
        }
    }

    @Test
    @DisplayName("Test Batch Norm Comp Graph Simple")
    void testBatchNormCompGraphSimple() {
        int numClasses = 2;
        int height = 3;
        int width = 3;
        int channels = 1;
        long seed = 123;
        int minibatchSize = 3;
        for (boolean useLogStd : new boolean[] { true, false }) {
            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).updater(new NoOp()).dataType(DataType.DOUBLE).weightInit(WeightInit.XAVIER).graphBuilder().addInputs("in").setInputTypes(InputType.convolutional(height, width, channels)).addLayer("bn", new BatchNormalization.Builder().useLogStd(useLogStd).build(), "in").addLayer("out", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nOut(numClasses).build(), "bn").setOutputs("out").build();
            ComputationGraph net = new ComputationGraph(conf);
            net.init();
            Random r = new Random(12345);
            // Order: examples, channels, height, width
            INDArray input = Nd4j.rand(new int[] { minibatchSize, channels, height, width });
            INDArray labels = Nd4j.zeros(minibatchSize, numClasses);
            for (int i = 0; i < minibatchSize; i++) {
                labels.putScalar(new int[] { i, r.nextInt(numClasses) }, 1.0);
            }
            // Mean and variance vars are not gradient checkable; mean/variance "gradient" is used to implement running mean/variance calc
            // i.e., runningMean = decay * runningMean + (1-decay) * batchMean
            // However, numerical gradient will be 0 as forward pass doesn't depend on this "parameter"
            Set<String> excludeParams = new HashSet<>(Arrays.asList("bn_mean", "bn_var"));
            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(net).inputs(new INDArray[] { input }).labels(new INDArray[] { labels }).excludeParams(excludeParams));
            assertTrue(gradOK);
            TestUtils.testModelSerialization(net);
        }
    }


}
