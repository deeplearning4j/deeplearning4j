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

package org.deeplearning4j.util;

import lombok.val;
import org.apache.commons.lang3.SerializationUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.*;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class ModelSerializerTest extends BaseDL4JTest {

    @Rule
    public TemporaryFolder tempDir = new TemporaryFolder();

    @Test
    public void testWriteMLNModel() throws Exception {
        int nIn = 5;
        int nOut = 6;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).l1(0.01)
                        .l2(0.01).updater(new Sgd(0.1)).activation(Activation.TANH).weightInit(WeightInit.XAVIER).list()
                        .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(20).build())
                        .layer(1, new DenseLayer.Builder().nIn(20).nOut(30).build()).layer(2, new OutputLayer.Builder()
                                        .lossFunction(LossFunctions.LossFunction.MSE).nIn(30).nOut(nOut).build())
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        File tempFile = tempDir.newFile();

        ModelSerializer.writeModel(net, tempFile, true);

        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(tempFile);

        assertEquals(network.getLayerWiseConfigurations().toJson(), net.getLayerWiseConfigurations().toJson());
        assertEquals(net.params(), network.params());
        assertEquals(net.getUpdater().getStateViewArray(), network.getUpdater().getStateViewArray());
    }

    @Test
    public void testWriteMlnModelInputStream() throws Exception {
        int nIn = 5;
        int nOut = 6;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).l1(0.01)
                        .l2(0.01).updater(new Sgd(0.1)).activation(Activation.TANH).weightInit(WeightInit.XAVIER).list()
                        .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(20).build())
                        .layer(1, new DenseLayer.Builder().nIn(20).nOut(30).build()).layer(2, new OutputLayer.Builder()
                                        .lossFunction(LossFunctions.LossFunction.MSE).nIn(30).nOut(nOut).build())
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        File tempFile = tempDir.newFile();
        FileOutputStream fos = new FileOutputStream(tempFile);

        ModelSerializer.writeModel(net, fos, true);


        // checking adding of DataNormalization to the model file

        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler();
        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        scaler.fit(iter);

        ModelSerializer.addNormalizerToModel(tempFile, scaler);

        NormalizerMinMaxScaler restoredScaler = ModelSerializer.restoreNormalizerFromFile(tempFile);

        assertNotEquals(null, scaler.getMax());
        assertEquals(scaler.getMax(), restoredScaler.getMax());
        assertEquals(scaler.getMin(), restoredScaler.getMin());

        FileInputStream fis = new FileInputStream(tempFile);

        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(fis);

        assertEquals(network.getLayerWiseConfigurations().toJson(), net.getLayerWiseConfigurations().toJson());
        assertEquals(net.params(), network.params());
        assertEquals(net.getUpdater().getStateViewArray(), network.getUpdater().getStateViewArray());
    }


    @Test
    public void testWriteCGModel() throws Exception {
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Sgd(0.1))
                        .graphBuilder().addInputs("in")
                        .addLayer("dense", new DenseLayer.Builder().nIn(4).nOut(2).build(), "in").addLayer("out",
                                        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(2).nOut(3)
                                                .activation(Activation.SOFTMAX).build(),
                                        "dense")
                        .setOutputs("out").build();

        ComputationGraph cg = new ComputationGraph(config);
        cg.init();

        File tempFile = tempDir.newFile();

        ModelSerializer.writeModel(cg, tempFile, true);

        ComputationGraph network = ModelSerializer.restoreComputationGraph(tempFile);

        assertEquals(network.getConfiguration().toJson(), cg.getConfiguration().toJson());
        assertEquals(cg.params(), network.params());
        assertEquals(cg.getUpdater().getStateViewArray(), network.getUpdater().getStateViewArray());
    }

    @Test
    public void testWriteCGModelInputStream() throws Exception {
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Sgd(0.1))
                        .graphBuilder().addInputs("in")
                        .addLayer("dense", new DenseLayer.Builder().nIn(4).nOut(2).build(), "in").addLayer("out",
                                        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(2).nOut(3)
                                                .activation(Activation.SOFTMAX).build(),
                                        "dense")
                        .setOutputs("out").build();

        ComputationGraph cg = new ComputationGraph(config);
        cg.init();

        File tempFile = tempDir.newFile();

        ModelSerializer.writeModel(cg, tempFile, true);
        FileInputStream fis = new FileInputStream(tempFile);

        ComputationGraph network = ModelSerializer.restoreComputationGraph(fis);

        assertEquals(network.getConfiguration().toJson(), cg.getConfiguration().toJson());
        assertEquals(cg.params(), network.params());
        assertEquals(cg.getUpdater().getStateViewArray(), network.getUpdater().getStateViewArray());
    }

    private DataSet trivialDataSet() {
        INDArray inputs = Nd4j.create(new float[] {1.0f, 2.0f, 3.0f});
        INDArray labels = Nd4j.create(new float[] {4.0f, 5.0f, 6.0f});
        return new DataSet(inputs, labels);
    }

    private ComputationGraph simpleComputationGraph() {
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Sgd(0.1))
                        .graphBuilder().addInputs("in")
                        .addLayer("dense", new DenseLayer.Builder().nIn(4).nOut(2).build(), "in").addLayer("out",
                                        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(2).nOut(3)
                                                .activation(Activation.SOFTMAX).build(),
                                        "dense")
                        .setOutputs("out").build();

        return new ComputationGraph(config);
    }

    @Test
    public void testSaveRestoreNormalizerFromInputStream() throws Exception {
        DataSet dataSet = trivialDataSet();
        NormalizerStandardize norm = new NormalizerStandardize();
        norm.fit(dataSet);

        ComputationGraph cg = simpleComputationGraph();
        cg.init();

        File tempFile = tempDir.newFile();

        ModelSerializer.writeModel(cg, tempFile, true);

        ModelSerializer.addNormalizerToModel(tempFile, norm);
        FileInputStream fis = new FileInputStream(tempFile);


        NormalizerStandardize restored = ModelSerializer.restoreNormalizerFromInputStream(fis);

        assertNotEquals(null, restored);

        DataSet dataSet2 = dataSet.copy();

        norm.preProcess(dataSet2);
        assertNotEquals(dataSet.getFeatures(), dataSet2.getFeatures());

        restored.revert(dataSet2);
        assertEquals(dataSet.getFeatures(), dataSet2.getFeatures());
    }

    @Test
    public void testRestoreUnsavedNormalizerFromInputStream() throws Exception {
        DataSet dataSet = trivialDataSet();

        NormalizerStandardize norm = new NormalizerStandardize();
        norm.fit(dataSet);

        ComputationGraph cg = simpleComputationGraph();
        cg.init();

        File tempFile = tempDir.newFile();
        ModelSerializer.writeModel(cg, tempFile, true);

        FileInputStream fis = new FileInputStream(tempFile);

        NormalizerStandardize restored = ModelSerializer.restoreNormalizerFromInputStream(fis);

        assertEquals(null, restored);
    }

    @Test
    public void testInvalidLoading1() throws Exception {
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .graphBuilder().addInputs("in")
                .addLayer("dense", new DenseLayer.Builder().nIn(4).nOut(2).build(), "in")
                .addLayer("out",new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX).nIn(2).nOut(3).build(),
                        "dense")
                .setOutputs("out").build();

        ComputationGraph cg = new ComputationGraph(config);
        cg.init();

        File tempFile = tempDir.newFile();

        ModelSerializer.writeModel(cg, tempFile, true);

        try {
            ModelSerializer.restoreMultiLayerNetwork(tempFile);
            fail();
        } catch (Exception e){
            String msg = e.getMessage();
            assertTrue(msg, msg.contains("JSON") && msg.contains("restoreComputationGraph"));
        }
    }

    @Test
    public void testInvalidLoading2() throws Exception {
        int nIn = 5;
        int nOut = 6;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).l1(0.01)
                .l2(0.01).updater(new Sgd(0.1)).activation(Activation.TANH).weightInit(WeightInit.XAVIER).list()
                .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(20).build())
                .layer(1, new DenseLayer.Builder().nIn(20).nOut(30).build()).layer(2, new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MSE).nIn(30).nOut(nOut).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        File tempFile = tempDir.newFile("testInvalidLoading2.bin");

        ModelSerializer.writeModel(net, tempFile, true);

        try {
            ModelSerializer.restoreComputationGraph(tempFile);
            fail();
        } catch (Exception e){
            String msg = e.getMessage();
            assertTrue(msg, msg.contains("JSON") && msg.contains("restoreMultiLayerNetwork"));
        }
    }

    @Test
    public void testInvalidStreamReuse() throws Exception {
        int nIn = 5;
        int nOut = 6;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).l1(0.01)
                .list()
                .layer(new OutputLayer.Builder().nIn(nIn).nOut(nOut).activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        DataSet dataSet = trivialDataSet();
        NormalizerStandardize norm = new NormalizerStandardize();
        norm.fit(dataSet);

        File tempFile = tempDir.newFile();
        ModelSerializer.writeModel(net, tempFile, true);
        ModelSerializer.addNormalizerToModel(tempFile, norm);

        InputStream is = new FileInputStream(tempFile);
        ModelSerializer.restoreMultiLayerNetwork(is);

        try{
            ModelSerializer.restoreNormalizerFromInputStream(is);
            fail("Expected exception");
        } catch (Exception e){
            String msg = e.getMessage();
            assertTrue(msg, msg.contains("may have been closed"));
        }

        try{
            ModelSerializer.restoreMultiLayerNetwork(is);
            fail("Expected exception");
        } catch (Exception e){
            String msg = e.getMessage();
            assertTrue(msg, msg.contains("may have been closed"));
        }

        //Also test reading  both model and normalizer from stream (correctly)
        Pair<MultiLayerNetwork,Normalizer> pair = ModelSerializer.restoreMultiLayerNetworkAndNormalizer(new FileInputStream(tempFile), true);
        assertEquals(net.params(), pair.getFirst().params());
        assertNotNull(pair.getSecond());
    }


    @Test
    public void testInvalidStreamReuseCG() throws Exception {
        int nIn = 5;
        int nOut = 6;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).l1(0.01)
                .graphBuilder()
                .addInputs("in")
                .layer("0", new OutputLayer.Builder().nIn(nIn).nOut(nOut).activation(Activation.SOFTMAX).build(), "in")
                .setOutputs("0")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        DataSet dataSet = trivialDataSet();
        NormalizerStandardize norm = new NormalizerStandardize();
        norm.fit(dataSet);

        File tempFile = tempDir.newFile();
        ModelSerializer.writeModel(net, tempFile, true);
        ModelSerializer.addNormalizerToModel(tempFile, norm);

        InputStream is = new FileInputStream(tempFile);
        ModelSerializer.restoreComputationGraph(is);

        try{
            ModelSerializer.restoreNormalizerFromInputStream(is);
            fail("Expected exception");
        } catch (Exception e){
            String msg = e.getMessage();
            assertTrue(msg, msg.contains("may have been closed"));
        }

        try{
            ModelSerializer.restoreComputationGraph(is);
            fail("Expected exception");
        } catch (Exception e){
            String msg = e.getMessage();
            assertTrue(msg, msg.contains("may have been closed"));
        }

        //Also test reading  both model and normalizer from stream (correctly)
        Pair<ComputationGraph,Normalizer> pair = ModelSerializer.restoreComputationGraphAndNormalizer(new FileInputStream(tempFile), true);
        assertEquals(net.params(), pair.getFirst().params());
        assertNotNull(pair.getSecond());
    }


    @Test
    public void testJavaSerde_1() throws Exception {
        int nIn = 5;
        int nOut = 6;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).l1(0.01)
                .graphBuilder()
                .addInputs("in")
                .layer("0", new OutputLayer.Builder().nIn(nIn).nOut(nOut).build(), "in")
                .setOutputs("0")
                .validateOutputLayerConfig(false)
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        DataSet dataSet = trivialDataSet();
        NormalizerStandardize norm = new NormalizerStandardize();
        norm.fit(dataSet);

        val b = SerializationUtils.serialize(net);

        ComputationGraph restored = SerializationUtils.deserialize(b);

        assertEquals(net, restored);
    }

    @Test
    public void testJavaSerde_2() throws Exception {
        int nIn = 5;
        int nOut = 6;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).l1(0.01)
                .list()
                .layer(0, new OutputLayer.Builder().nIn(nIn).nOut(nOut).activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        DataSet dataSet = trivialDataSet();
        NormalizerStandardize norm = new NormalizerStandardize();
        norm.fit(dataSet);

        val b = SerializationUtils.serialize(net);

        MultiLayerNetwork restored = SerializationUtils.deserialize(b);

        assertEquals(net, restored);
    }

    @Test
    public void testPutGetObject() throws Exception {

        int nIn = 5;
        int nOut = 6;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).l1(0.01)
                .graphBuilder()
                .addInputs("in")
                .layer("0", new OutputLayer.Builder().nIn(nIn).nOut(nOut).activation(Activation.SOFTMAX).build(), "in")
                .setOutputs("0")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        File tempFile = tempDir.newFile();
        ModelSerializer.writeModel(net, tempFile, true);


        List<String> toWrite = Arrays.asList("zero", "one", "two");
        ModelSerializer.addObjectToFile(tempFile, "myLabels", toWrite);
        List<String> restored = ModelSerializer.getObjectFromFile(tempFile, "myLabels");
        assertEquals(toWrite, restored);


        Map<String,Object> someOtherData = new HashMap<>();
        someOtherData.put("x", new float[]{0,1,2});
        someOtherData.put("y",Nd4j.linspace(1,10,10));

        ModelSerializer.addObjectToFile(tempFile, "otherData.bin", someOtherData);

        Map<String,Object> dataRestored = ModelSerializer.getObjectFromFile(tempFile, "otherData.bin");
        assertEquals(someOtherData.keySet(), dataRestored.keySet());
        assertArrayEquals((float[])someOtherData.get("x"), (float[])dataRestored.get("x"), 0f);
        assertEquals(someOtherData.get("y"), dataRestored.get("y"));


        List<String> entries = ModelSerializer.listObjectsInFile(tempFile);
        assertEquals(2, entries.size());
        System.out.println(entries);
        assertTrue(entries.contains("myLabels"));
        assertTrue(entries.contains("otherData.bin"));

        ComputationGraph restoredNet = ModelSerializer.restoreComputationGraph(tempFile);
        assertEquals(net.params(), restoredNet.params());
    }
}
