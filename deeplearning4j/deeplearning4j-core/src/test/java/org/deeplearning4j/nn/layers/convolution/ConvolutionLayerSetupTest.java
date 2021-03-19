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
package org.deeplearning4j.nn.layers.convolution;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.FeatureUtil;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.DisplayName;
import java.nio.file.Path;
import org.junit.jupiter.api.extension.ExtendWith;

/**
 * @author Adam Gibson
 */
@DisplayName("Convolution Layer Setup Test")
class ConvolutionLayerSetupTest extends BaseDL4JTest {

    @TempDir
    public Path testDir;

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    @Test
    @DisplayName("Test Convolution Layer Setup")
    void testConvolutionLayerSetup() {
        MultiLayerConfiguration.Builder builder = inComplete();
        builder.setInputType(InputType.convolutionalFlat(28, 28, 1));
        MultiLayerConfiguration completed = complete().build();
        MultiLayerConfiguration test = builder.build();
        assertEquals(completed, test);
    }

    @Test
    @DisplayName("Test Dense To Output Layer")
    void testDenseToOutputLayer() {
        Nd4j.getRandom().setSeed(12345);
        final int numRows = 76;
        final int numColumns = 76;
        int nChannels = 3;
        int outputNum = 6;
        int seed = 123;
        // setup the network
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(seed).l1(1e-1).l2(2e-4).dropOut(0.5).miniBatch(true).optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).list().layer(0, new ConvolutionLayer.Builder(5, 5).nOut(5).dropOut(0.5).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build()).layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] { 2, 2 }).build()).layer(2, new ConvolutionLayer.Builder(3, 3).nOut(10).dropOut(0.5).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build()).layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] { 2, 2 }).build()).layer(4, new DenseLayer.Builder().nOut(100).activation(Activation.RELU).build()).layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build()).setInputType(InputType.convolutional(numRows, numColumns, nChannels));
        DataSet d = new DataSet(Nd4j.rand(new int[] { 10, nChannels, numRows, numColumns }), FeatureUtil.toOutcomeMatrix(new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, 6));
        MultiLayerNetwork network = new MultiLayerNetwork(builder.build());
        network.init();
        network.fit(d);
    }

    @Test
    @DisplayName("Test Mnist Lenet")
    void testMnistLenet() throws Exception {
        MultiLayerConfiguration.Builder incomplete = incompleteMnistLenet();
        incomplete.setInputType(InputType.convolutionalFlat(28, 28, 1));
        MultiLayerConfiguration testConf = incomplete.build();
        assertEquals(800, ((FeedForwardLayer) testConf.getConf(4).getLayer()).getNIn());
        assertEquals(500, ((FeedForwardLayer) testConf.getConf(5).getLayer()).getNIn());
        // test instantiation
        DataSetIterator iter = new MnistDataSetIterator(10, 10);
        MultiLayerNetwork network = new MultiLayerNetwork(testConf);
        network.init();
        network.fit(iter.next());
    }

    @Test
    @DisplayName("Test Multi Channel")
    void testMultiChannel() throws Exception {
        INDArray in = Nd4j.rand(new int[] { 10, 3, 28, 28 });
        INDArray labels = Nd4j.rand(10, 2);
        DataSet next = new DataSet(in, labels);
        NeuralNetConfiguration.ListBuilder builder = (NeuralNetConfiguration.ListBuilder) incompleteLFW();
        builder.setInputType(InputType.convolutional(28, 28, 3));
        MultiLayerConfiguration conf = builder.build();
        ConvolutionLayer layer2 = (ConvolutionLayer) conf.getConf(2).getLayer();
        assertEquals(6, layer2.getNIn());
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.fit(next);
    }

    @Test
    @DisplayName("Test LRN")
    void testLRN(@TempDir Path testFolder) throws Exception {
        List<String> labels = new ArrayList<>(Arrays.asList("Zico", "Ziwang_Xu"));
        File dir = testFolder.toFile();
        new ClassPathResource("lfwtest/").copyDirectory(dir);
        String rootDir = dir.getAbsolutePath();
        RecordReader reader = new ImageRecordReader(28, 28, 3);
        reader.initialize(new FileSplit(new File(rootDir)));
        DataSetIterator recordReader = new RecordReaderDataSetIterator(reader, 10, 1, labels.size());
        labels.remove("lfwtest");
        NeuralNetConfiguration.ListBuilder builder = (NeuralNetConfiguration.ListBuilder) incompleteLRN();
        builder.setInputType(InputType.convolutional(28, 28, 3));
        MultiLayerConfiguration conf = builder.build();
        ConvolutionLayer layer2 = (ConvolutionLayer) conf.getConf(3).getLayer();
        assertEquals(6, layer2.getNIn());
    }

    public MultiLayerConfiguration.Builder incompleteLRN() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(3).optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).list().layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[] { 5, 5 }).nOut(6).build()).layer(1, new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[] { 2, 2 }).build()).layer(2, new LocalResponseNormalization.Builder().build()).layer(3, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[] { 5, 5 }).nOut(6).build()).layer(4, new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[] { 2, 2 }).build()).layer(5, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(2).activation(Activation.SOFTMAX).build());
        return builder;
    }

    public MultiLayerConfiguration.Builder incompleteLFW() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(3).optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).list().layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[] { 5, 5 }).nOut(6).build()).layer(1, new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[] { 2, 2 }).build()).layer(2, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[] { 5, 5 }).nOut(6).build()).layer(3, new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[] { 2, 2 }).build()).layer(4, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nOut(2).build());
        return builder;
    }

    public MultiLayerConfiguration.Builder incompleteMnistLenet() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(3).optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).list().layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[] { 5, 5 }).nIn(1).nOut(20).build()).layer(1, new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[] { 2, 2 }, new int[] { 2, 2 }).build()).layer(2, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[] { 5, 5 }).nIn(20).nOut(50).build()).layer(3, new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[] { 2, 2 }, new int[] { 2, 2 }).build()).layer(4, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nOut(500).build()).layer(5, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nOut(10).build());
        return builder;
    }

    public MultiLayerConfiguration mnistLenet() {
        MultiLayerConfiguration builder = new NeuralNetConfiguration.Builder().seed(3).optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).list().layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[] { 5, 5 }).nIn(1).nOut(6).build()).layer(1, new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[] { 5, 5 }, new int[] { 2, 2 }).build()).layer(2, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[] { 5, 5 }).nIn(1).nOut(6).build()).layer(3, new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[] { 5, 5 }, new int[] { 2, 2 }).build()).layer(4, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(150).nOut(10).build()).build();
        return builder;
    }

    public MultiLayerConfiguration.Builder inComplete() {
        int nChannels = 1;
        int outputNum = 10;
        int seed = 123;
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT).list().layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[] { 10, 10 }, new int[] { 2, 2 }).nIn(nChannels).nOut(6).build()).layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] { 2, 2 }).build()).layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build());
        return builder;
    }

    public MultiLayerConfiguration.Builder complete() {
        final int numRows = 28;
        final int numColumns = 28;
        int nChannels = 1;
        int outputNum = 10;
        int seed = 123;
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT).list().layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[] { 10, 10 }, new int[] { 2, 2 }).nIn(nChannels).nOut(6).build()).layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] { 2, 2 }).build()).layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(// 216
        5 * 5 * 1 * 6).nOut(outputNum).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build()).inputPreProcessor(0, new FeedForwardToCnnPreProcessor(numRows, numColumns, nChannels)).inputPreProcessor(2, new CnnToFeedForwardPreProcessor(5, 5, 6));
        return builder;
    }

    @Test
    @DisplayName("Test Deconvolution")
    void testDeconvolution() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().list().layer(0, new Deconvolution2D.Builder(2, 2).padding(0, 0).stride(2, 2).nIn(1).nOut(3).build()).layer(1, new SubsamplingLayer.Builder().kernelSize(2, 2).padding(1, 1).stride(2, 2).build()).layer(2, new OutputLayer.Builder().nOut(3).activation(Activation.SOFTMAX).build()).setInputType(InputType.convolutional(28, 28, 1));
        MultiLayerConfiguration conf = builder.build();
        assertNotNull(conf.getInputPreProcess(2));
        assertTrue(conf.getInputPreProcess(2) instanceof CnnToFeedForwardPreProcessor);
        CnnToFeedForwardPreProcessor proc = (CnnToFeedForwardPreProcessor) conf.getInputPreProcess(2);
        assertEquals(29, proc.getInputHeight());
        assertEquals(29, proc.getInputWidth());
        assertEquals(3, proc.getNumChannels());
        assertEquals(29 * 29 * 3, ((FeedForwardLayer) conf.getConf(2).getLayer()).getNIn());
    }

    @Test
    @DisplayName("Test Sub Sampling With Padding")
    void testSubSamplingWithPadding() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().list().layer(0, // (28-2+0)/2+1 = 14
        new ConvolutionLayer.Builder(2, 2).padding(0, 0).stride(2, 2).nIn(1).nOut(3).build()).layer(1, // (14-2+2)/2+1 = 8 -> 8x8x3
        new SubsamplingLayer.Builder().kernelSize(2, 2).padding(1, 1).stride(2, 2).build()).layer(2, new OutputLayer.Builder().nOut(3).activation(Activation.SOFTMAX).build()).setInputType(InputType.convolutional(28, 28, 1));
        MultiLayerConfiguration conf = builder.build();
        assertNotNull(conf.getInputPreProcess(2));
        assertTrue(conf.getInputPreProcess(2) instanceof CnnToFeedForwardPreProcessor);
        CnnToFeedForwardPreProcessor proc = (CnnToFeedForwardPreProcessor) conf.getInputPreProcess(2);
        assertEquals(8, proc.getInputHeight());
        assertEquals(8, proc.getInputWidth());
        assertEquals(3, proc.getNumChannels());
        assertEquals(8 * 8 * 3, ((FeedForwardLayer) conf.getConf(2).getLayer()).getNIn());
    }

    @Test
    @DisplayName("Test Upsampling")
    void testUpsampling() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().list().layer(// (28-2+0)/2+1 = 14
        new ConvolutionLayer.Builder(2, 2).padding(0, 0).stride(2, 2).nIn(1).nOut(3).build()).layer(// 14 * 3 = 42!
        new Upsampling2D.Builder().size(3).build()).layer(new OutputLayer.Builder().nOut(3).activation(Activation.SOFTMAX).build()).setInputType(InputType.convolutional(28, 28, 1));
        MultiLayerConfiguration conf = builder.build();
        assertNotNull(conf.getInputPreProcess(2));
        assertTrue(conf.getInputPreProcess(2) instanceof CnnToFeedForwardPreProcessor);
        CnnToFeedForwardPreProcessor proc = (CnnToFeedForwardPreProcessor) conf.getInputPreProcess(2);
        assertEquals(42, proc.getInputHeight());
        assertEquals(42, proc.getInputWidth());
        assertEquals(3, proc.getNumChannels());
        assertEquals(42 * 42 * 3, ((FeedForwardLayer) conf.getConf(2).getLayer()).getNIn());
    }

    @Test
    @DisplayName("Test Space To Batch")
    void testSpaceToBatch() {
        int[] blocks = new int[] { 2, 2 };
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().list().layer(// (28-2+0)/2+1 = 14
        new ConvolutionLayer.Builder(2, 2).padding(0, 0).stride(2, 2).nIn(1).nOut(3).build()).layer(// Divide space dimensions by blocks, i.e. 14/2 = 7
        new SpaceToBatchLayer.Builder(blocks).build()).layer(new OutputLayer.Builder().nOut(3).activation(Activation.SOFTMAX).build()).setInputType(InputType.convolutional(28, 28, 1));
        MultiLayerConfiguration conf = builder.build();
        assertNotNull(conf.getInputPreProcess(2));
        assertTrue(conf.getInputPreProcess(2) instanceof CnnToFeedForwardPreProcessor);
        CnnToFeedForwardPreProcessor proc = (CnnToFeedForwardPreProcessor) conf.getInputPreProcess(2);
        assertEquals(7, proc.getInputHeight());
        assertEquals(7, proc.getInputWidth());
        assertEquals(3, proc.getNumChannels());
    }

    @Test
    @DisplayName("Test Space To Depth")
    void testSpaceToDepth() {
        int blocks = 2;
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().list().layer(new ConvolutionLayer.Builder(2, 2).padding(0, 0).stride(2, 2).nIn(1).nOut(3).build()).layer(new SpaceToDepthLayer.Builder(blocks, SpaceToDepthLayer.DataFormat.NCHW).build()).layer(// nIn of the next layer gets multiplied by 2*2.
        new OutputLayer.Builder().nIn(3 * 2 * 2).nOut(3).activation(Activation.SOFTMAX).build()).setInputType(InputType.convolutional(28, 28, 1));
        MultiLayerConfiguration conf = builder.build();
        assertNotNull(conf.getInputPreProcess(2));
        assertTrue(conf.getInputPreProcess(2) instanceof CnnToFeedForwardPreProcessor);
        CnnToFeedForwardPreProcessor proc = (CnnToFeedForwardPreProcessor) conf.getInputPreProcess(2);
        assertEquals(7, proc.getInputHeight());
        assertEquals(7, proc.getInputWidth());
        assertEquals(12, proc.getNumChannels());
    }

    @Test
    @DisplayName("Test CNNDBN Multi Layer")
    void testCNNDBNMultiLayer() throws Exception {
        DataSetIterator iter = new MnistDataSetIterator(2, 2);
        DataSet next = iter.next();
        // Run with separate activation layer
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(123).weightInit(WeightInit.XAVIER).list().layer(0, new ConvolutionLayer.Builder(new int[] { 1, 1 }, new int[] { 1, 1 }).nIn(1).nOut(6).activation(Activation.IDENTITY).build()).layer(1, new BatchNormalization.Builder().build()).layer(2, new ActivationLayer.Builder().activation(Activation.RELU).build()).layer(3, new DenseLayer.Builder().nIn(28 * 28 * 6).nOut(10).activation(Activation.IDENTITY).build()).layer(4, new BatchNormalization.Builder().nOut(10).build()).layer(5, new ActivationLayer.Builder().activation(Activation.RELU).build()).layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nOut(10).build()).setInputType(InputType.convolutionalFlat(28, 28, 1)).build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setInput(next.getFeatures());
        INDArray activationsActual = network.output(next.getFeatures());
        assertEquals(10, activationsActual.shape()[1], 1e-2);
        network.fit(next);
        INDArray actualGammaParam = network.getLayer(1).getParam(BatchNormalizationParamInitializer.GAMMA);
        INDArray actualBetaParam = network.getLayer(1).getParam(BatchNormalizationParamInitializer.BETA);
        assertTrue(actualGammaParam != null);
        assertTrue(actualBetaParam != null);
    }

    @Test
    @DisplayName("Test Separable Conv 2 D")
    void testSeparableConv2D() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().list().layer(new SeparableConvolution2D.Builder(2, 2).depthMultiplier(2).padding(0, 0).stride(2, 2).nIn(1).nOut(3).build()).layer(// (14-2+2)/2+1 = 8 -> 8x8x3
        new SubsamplingLayer.Builder().kernelSize(2, 2).padding(1, 1).stride(2, 2).build()).layer(2, new OutputLayer.Builder().nOut(3).activation(Activation.SOFTMAX).build()).setInputType(InputType.convolutional(28, 28, 1));
        MultiLayerConfiguration conf = builder.build();
        assertNotNull(conf.getInputPreProcess(2));
        assertTrue(conf.getInputPreProcess(2) instanceof CnnToFeedForwardPreProcessor);
        CnnToFeedForwardPreProcessor proc = (CnnToFeedForwardPreProcessor) conf.getInputPreProcess(2);
        assertEquals(8, proc.getInputHeight());
        assertEquals(8, proc.getInputWidth());
        assertEquals(3, proc.getNumChannels());
        assertEquals(8 * 8 * 3, ((FeedForwardLayer) conf.getConf(2).getLayer()).getNIn());
    }

    @Test
    @DisplayName("Test Deconv 2 D")
    void testDeconv2D() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().list().layer(new Deconvolution2D.Builder(2, 2).padding(0, 0).stride(2, 2).nIn(1).nOut(3).build()).layer(new SubsamplingLayer.Builder().kernelSize(2, 2).padding(1, 1).stride(2, 2).build()).layer(2, new OutputLayer.Builder().nOut(3).activation(Activation.SOFTMAX).build()).setInputType(InputType.convolutional(28, 28, 1));
        MultiLayerConfiguration conf = builder.build();
        assertNotNull(conf.getInputPreProcess(2));
        assertTrue(conf.getInputPreProcess(2) instanceof CnnToFeedForwardPreProcessor);
        CnnToFeedForwardPreProcessor proc = (CnnToFeedForwardPreProcessor) conf.getInputPreProcess(2);
        assertEquals(29, proc.getInputHeight());
        assertEquals(29, proc.getInputWidth());
        assertEquals(3, proc.getNumChannels());
        assertEquals(29 * 29 * 3, ((FeedForwardLayer) conf.getConf(2).getLayer()).getNIn());
    }
}
