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
package org.eclipse.deeplearning4j.dl4jcore.nn.conf;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ListBuilder;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.weightnoise.DropConnect;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.*;
import java.util.Arrays;
import java.util.Properties;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.DisplayName;
import java.nio.file.Path;

@Slf4j
@DisplayName("Multi Layer Neural Net Configuration Test")
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
class MultiLayerNeuralNetConfigurationTest extends BaseDL4JTest {

    @TempDir
    public Path testDir;

    @Test
    @DisplayName("Test Json")
    void testJson() throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list().layer(0, new DenseLayer.Builder().dist(new NormalDistribution(1, 1e-1)).build()).inputPreProcessor(0, new CnnToFeedForwardPreProcessor()).build();
        String json = conf.toJson();
        MultiLayerConfiguration from = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf.getConf(0), from.getConf(0));
        Properties props = new Properties();
        props.put("json", json);
        String key = props.getProperty("json");
        assertEquals(json, key);
        File f = testDir.resolve("props").toFile();
        f.deleteOnExit();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(f));
        props.store(bos, "");
        bos.flush();
        bos.close();
        BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f));
        Properties props2 = new Properties();
        props2.load(bis);
        bis.close();
        assertEquals(props2.getProperty("json"), props.getProperty("json"));
        String json2 = props2.getProperty("json");
        MultiLayerConfiguration conf3 = MultiLayerConfiguration.fromJson(json2);
        assertEquals(conf.getConf(0), conf3.getConf(0));
    }

    @Test
    @DisplayName("Test Convnet Json")
    void testConvnetJson() {
        final int numRows = 76;
        final int numColumns = 76;
        int nChannels = 3;
        int outputNum = 6;
        int seed = 123;
        // setup the network
        ListBuilder builder = new NeuralNetConfiguration.Builder().seed(seed).l1(1e-1).l2(2e-4)
                .weightNoise(new DropConnect(0.5)).miniBatch(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list().layer(0, new ConvolutionLayer.Builder(5, 5).nOut(5)
                        .dropOut(0.5)
                        .weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] { 2, 2 }).build())
                .layer(2, new ConvolutionLayer.Builder(3, 3).nOut(10).dropOut(0.5)
                        .weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] { 2, 2 }).build())
                .layer(4, new DenseLayer.Builder().nOut(100).activation(Activation.RELU).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build())
                .setInputType(InputType.convolutional(numRows, numColumns, nChannels));
        MultiLayerConfiguration conf = builder.build();
        String json = conf.toJson();
        MultiLayerConfiguration conf2 = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf, conf2);
    }

    @Test
    @DisplayName("Test Upsampling Convnet Json")
    void testUpsamplingConvnetJson() {
        final int numRows = 76;
        final int numColumns = 76;
        int nChannels = 3;
        int outputNum = 6;
        int seed = 123;
        // setup the network
        ListBuilder builder = new NeuralNetConfiguration.Builder().seed(seed).l1(1e-1).l2(2e-4).dropOut(0.5)
                .miniBatch(true).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list().layer(new ConvolutionLayer.Builder(5, 5).nOut(5)
                        .dropOut(0.5).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
                .layer(new Upsampling2D.Builder().size(2).build()).layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nOut(10).dropOut(0.5).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
                .layer(new Upsampling2D.Builder().size(2).build()).layer(4, new DenseLayer.Builder().nOut(100)
                        .activation(Activation.RELU).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum)
                        .weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build())
                .setInputType(InputType.convolutional(numRows, numColumns, nChannels));
        MultiLayerConfiguration conf = builder.build();
        String json = conf.toJson();
        MultiLayerConfiguration conf2 = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf, conf2);
    }

    @Test
    @DisplayName("Test Global Pooling Json")
    void testGlobalPoolingJson() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new NoOp()).dist(new NormalDistribution(0, 1.0)).seed(12345L).list().layer(0, new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).nOut(5).build()).layer(1, new GlobalPoolingLayer.Builder().poolingType(PoolingType.PNORM).pnorm(3).build()).layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nOut(3).build()).setInputType(InputType.convolutional(32, 32, 1)).build();
        String str = conf.toJson();
        MultiLayerConfiguration fromJson = conf.fromJson(str);
        assertEquals(conf, fromJson);
    }

    @Test
    @DisplayName("Test Yaml")
    void testYaml() throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list().layer(0, new DenseLayer.Builder().dist(new NormalDistribution(1, 1e-1)).build()).inputPreProcessor(0, new CnnToFeedForwardPreProcessor()).build();
        String json = conf.toYaml();
        MultiLayerConfiguration from = MultiLayerConfiguration.fromYaml(json);
        assertEquals(conf.getConf(0), from.getConf(0));
        Properties props = new Properties();
        props.put("json", json);
        String key = props.getProperty("json");
        assertEquals(json, key);
        File f = testDir.resolve("props").toFile();
        f.deleteOnExit();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(f));
        props.store(bos, "");
        bos.flush();
        bos.close();
        BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f));
        Properties props2 = new Properties();
        props2.load(bis);
        bis.close();
        assertEquals(props2.getProperty("json"), props.getProperty("json"));
        String yaml = props2.getProperty("json");
        MultiLayerConfiguration conf3 = MultiLayerConfiguration.fromYaml(yaml);
        assertEquals(conf.getConf(0), conf3.getConf(0));
    }

    @Test
    @DisplayName("Test Clone")
    void testClone() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list().layer(0, new DenseLayer.Builder().build()).layer(1, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).build()).inputPreProcessor(1, new CnnToFeedForwardPreProcessor()).build();
        MultiLayerConfiguration conf2 = conf.clone();
        assertEquals(conf, conf2);
        assertNotSame(conf, conf2);
        assertNotSame(conf.getConfs(), conf2.getConfs());
        for (int i = 0; i < conf.getConfs().size(); i++) {
            assertNotSame(conf.getConf(i), conf2.getConf(i));
        }
        assertNotSame(conf.getInputPreProcessors(), conf2.getInputPreProcessors());
        for (Integer layer : conf.getInputPreProcessors().keySet()) {
            assertNotSame(conf.getInputPreProcess(layer), conf2.getInputPreProcess(layer));
        }
    }

    @Test
    @DisplayName("Test Random Weight Init")
    void testRandomWeightInit() {
        MultiLayerNetwork model1 = new MultiLayerNetwork(getConf());
        model1.init();
        Nd4j.getRandom().setSeed(12345L);
        MultiLayerNetwork model2 = new MultiLayerNetwork(getConf());
        model2.init();
        float[] p1 = model1.params().data().asFloat();
        float[] p2 = model2.params().data().asFloat();
        System.out.println(Arrays.toString(p1));
        System.out.println(Arrays.toString(p2));
        assertArrayEquals(p1, p2, 0.0f);
    }

    @Test
    @DisplayName("Test Training Listener")
    void testTrainingListener() {
        MultiLayerNetwork model1 = new MultiLayerNetwork(getConf());
        model1.init();
        model1.addListeners(new ScoreIterationListener(1));
        MultiLayerNetwork model2 = new MultiLayerNetwork(getConf());
        model2.addListeners(new ScoreIterationListener(1));
        model2.init();
        Layer[] l1 = model1.getLayers();
        for (int i = 0; i < l1.length; i++) assertTrue(l1[i].getListeners() != null && l1[i].getListeners().size() == 1);
        Layer[] l2 = model2.getLayers();
        for (int i = 0; i < l2.length; i++) assertTrue(l2[i].getListeners() != null && l2[i].getListeners().size() == 1);
    }

    private static MultiLayerConfiguration getConf() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345l).list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).dist(new NormalDistribution(0, 1)).build()).layer(1, new OutputLayer.Builder().nIn(2).nOut(1).activation(Activation.TANH).dist(new NormalDistribution(0, 1)).lossFunction(LossFunctions.LossFunction.MSE).build()).build();
        return conf;
    }

    @Test
    @DisplayName("Test Invalid Config")
    void testInvalidConfig() {
        try {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).list().build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            fail("No exception thrown for invalid configuration");
        } catch (IllegalStateException e) {
            // OK
            log.error("", e);
        } catch (Throwable e) {
            log.error("", e);
            fail("Unexpected exception thrown for invalid config");
        }
        try {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).list().layer(1, new DenseLayer.Builder().nIn(3).nOut(4).build()).layer(2, new OutputLayer.Builder().nIn(4).nOut(5).build()).build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            fail("No exception thrown for invalid configuration");
        } catch (IllegalStateException e) {
            // OK
            log.info(e.toString());
        } catch (Throwable e) {
            log.error("", e);
            fail("Unexpected exception thrown for invalid config");
        }
        try {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).list().layer(0, new DenseLayer.Builder().nIn(3).nOut(4).build()).layer(2, new OutputLayer.Builder().nIn(4).nOut(5).build()).build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            fail("No exception thrown for invalid configuration");
        } catch (IllegalStateException e) {
            // OK
            log.info(e.toString());
        } catch (Throwable e) {
            log.error("", e);
            fail("Unexpected exception thrown for invalid config");
        }
    }

    @Test
    @DisplayName("Test List Overloads")
    void testListOverloads() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).list().layer(0, new DenseLayer.Builder().nIn(3).nOut(4).build()).layer(1, new OutputLayer.Builder().nIn(4).nOut(5).activation(Activation.SOFTMAX).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        DenseLayer dl = (DenseLayer) conf.getConf(0).getLayer();
        assertEquals(3, dl.getNIn());
        assertEquals(4, dl.getNOut());
        OutputLayer ol = (OutputLayer) conf.getConf(1).getLayer();
        assertEquals(4, ol.getNIn());
        assertEquals(5, ol.getNOut());
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder().seed(12345).list().layer(0, new DenseLayer.Builder().nIn(3).nOut(4).build()).layer(1, new OutputLayer.Builder().nIn(4).nOut(5).activation(Activation.SOFTMAX).build()).build();
        MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
        net2.init();
        MultiLayerConfiguration conf3 = new NeuralNetConfiguration.Builder().seed(12345).list(new DenseLayer.Builder().nIn(3).nOut(4).build(), new OutputLayer.Builder().nIn(4).nOut(5).activation(Activation.SOFTMAX).build()).build();
        MultiLayerNetwork net3 = new MultiLayerNetwork(conf3);
        net3.init();
        assertEquals(conf, conf2);
        assertEquals(conf, conf3);
    }

    @Test
    @DisplayName("Test Bias Lr")
    void testBiasLr() {
        // setup the network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).updater(new Adam(1e-2)).biasUpdater(new Adam(0.5)).list().layer(0, new ConvolutionLayer.Builder(5, 5).nOut(5).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build()).layer(1, new DenseLayer.Builder().nOut(100).activation(Activation.RELU).build()).layer(2, new DenseLayer.Builder().nOut(100).activation(Activation.RELU).build()).layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(10).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build()).setInputType(InputType.convolutional(28, 28, 1)).build();
        BaseLayer l0 = (BaseLayer) conf.getConf(0).getLayer();
        BaseLayer l1 = (BaseLayer) conf.getConf(1).getLayer();
        BaseLayer l2 = (BaseLayer) conf.getConf(2).getLayer();
        BaseLayer l3 = (BaseLayer) conf.getConf(3).getLayer();
        assertEquals(0.5, ((Adam) l0.getUpdaterByParam("b")).getLearningRate(), 1e-6);
        assertEquals(1e-2, ((Adam) l0.getUpdaterByParam("W")).getLearningRate(), 1e-6);
        assertEquals(0.5, ((Adam) l1.getUpdaterByParam("b")).getLearningRate(), 1e-6);
        assertEquals(1e-2, ((Adam) l1.getUpdaterByParam("W")).getLearningRate(), 1e-6);
        assertEquals(0.5, ((Adam) l2.getUpdaterByParam("b")).getLearningRate(), 1e-6);
        assertEquals(1e-2, ((Adam) l2.getUpdaterByParam("W")).getLearningRate(), 1e-6);
        assertEquals(0.5, ((Adam) l3.getUpdaterByParam("b")).getLearningRate(), 1e-6);
        assertEquals(1e-2, ((Adam) l3.getUpdaterByParam("W")).getLearningRate(), 1e-6);
    }

    @Test
    @DisplayName("Test Invalid Output Layer")
    void testInvalidOutputLayer() {
        /*
        Test case (invalid configs)
        1. nOut=1 + softmax
        2. mcxent + tanh
        3. xent + softmax
        4. xent + relu
        5. mcxent + sigmoid
         */
        LossFunctions.LossFunction[] lf = new LossFunctions.LossFunction[] { LossFunctions.LossFunction.MCXENT, LossFunctions.LossFunction.MCXENT, LossFunctions.LossFunction.XENT, LossFunctions.LossFunction.XENT, LossFunctions.LossFunction.MCXENT };
        int[] nOut = new int[] { 1, 3, 3, 3, 3 };
        Activation[] activations = new Activation[] { Activation.SOFTMAX, Activation.TANH, Activation.SOFTMAX, Activation.RELU, Activation.SIGMOID };
        for (int i = 0; i < lf.length; i++) {
            for (boolean lossLayer : new boolean[] { false, true }) {
                for (boolean validate : new boolean[] { true, false }) {
                    String s = "nOut=" + nOut[i] + ",lossFn=" + lf[i] + ",lossLayer=" + lossLayer + ",validate=" + validate;
                    if (nOut[i] == 1 && lossLayer)
                        // nOuts are not availabel in loss layer, can't expect it to detect this case
                        continue;
                    try {
                        new NeuralNetConfiguration.Builder().list().layer(new DenseLayer.Builder().nIn(10).nOut(10).build()).layer(!lossLayer ? new OutputLayer.Builder().nIn(10).nOut(nOut[i]).activation(activations[i]).lossFunction(lf[i]).build() : new LossLayer.Builder().activation(activations[i]).lossFunction(lf[i]).build()).validateOutputLayerConfig(validate).build();
                        if (validate) {
                            fail("Expected exception: " + s);
                        }
                    } catch (DL4JInvalidConfigException e) {
                        if (validate) {
                            assertTrue(e.getMessage().toLowerCase().contains("invalid output"),s);
                        } else {
                            fail("Validation should not be enabled");
                        }
                    }
                }
            }
        }
    }
}
