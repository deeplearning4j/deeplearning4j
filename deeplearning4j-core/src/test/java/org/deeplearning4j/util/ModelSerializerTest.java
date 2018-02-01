package org.deeplearning4j.util;

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
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * @author raver119@gmail.com
 */
public class ModelSerializerTest extends BaseDL4JTest {

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

        File tempFile = File.createTempFile("tsfs", "fdfsdf");
        tempFile.deleteOnExit();

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

        File tempFile = File.createTempFile("tsfs", "fdfsdf");
        tempFile.deleteOnExit();
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
                                                        .build(),
                                        "dense")
                        .setOutputs("out").pretrain(false).backprop(true).build();

        ComputationGraph cg = new ComputationGraph(config);
        cg.init();

        File tempFile = File.createTempFile("tsfs", "fdfsdf");
        tempFile.deleteOnExit();

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
                                                        .build(),
                                        "dense")
                        .setOutputs("out").pretrain(false).backprop(true).build();

        ComputationGraph cg = new ComputationGraph(config);
        cg.init();

        File tempFile = File.createTempFile("tsfs", "fdfsdf");
        tempFile.deleteOnExit();

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
                                                        .build(),
                                        "dense")
                        .setOutputs("out").pretrain(false).backprop(true).build();

        return new ComputationGraph(config);
    }

    @Test
    public void testSaveRestoreNormalizerFromInputStream() throws Exception {
        DataSet dataSet = trivialDataSet();

        NormalizerStandardize norm = new NormalizerStandardize();
        norm.fit(dataSet);

        ComputationGraph cg = simpleComputationGraph();
        cg.init();

        File tempFile = File.createTempFile("tsfs", "fdfsdf");
        tempFile.deleteOnExit();

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

        File tempFile = File.createTempFile("tsfs", "fdfsdf");
        tempFile.deleteOnExit();
        ModelSerializer.writeModel(cg, tempFile, true);

        FileInputStream fis = new FileInputStream(tempFile);

        NormalizerStandardize restored = ModelSerializer.restoreNormalizerFromInputStream(fis);

        assertEquals(null, restored);
    }
}
