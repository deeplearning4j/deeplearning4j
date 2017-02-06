package org.deeplearning4j.nn.multilayer;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by susaneraly on 2/4/17.
 */
public class MultiLayerTransferLearningTest {

    private static final Logger log = LoggerFactory.getLogger(MultiLayerTransferLearningTest.class);


    /*
        FineTune overrides all the learning related configs set in the model conf with the new ones specified
     */
    @Test
    public void simpleFineTune() {

        long rng = 12345L;
        DataSet randomData = new DataSet(Nd4j.rand(10,4),Nd4j.rand(10,3));
        MultiLayerConfiguration confToChange = new NeuralNetConfiguration.Builder()
                .seed(rng)
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .updater(Updater.NESTEROVS).momentum(0.99)
                .learningRate(0.01)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4).nOut(3)
                        .build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(3)
                        .build())
                .build();

        MultiLayerConfiguration confToSet = new NeuralNetConfiguration.Builder()
                .seed(rng)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP)
                .learningRate(0.0001)
                .regularization(true)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4).nOut(3)
                        .build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(3)
                        .build())
                .build();
        MultiLayerNetwork expectedModel = new MultiLayerNetwork(confToSet);
        expectedModel.init();
        MultiLayerConfiguration expectedConf = expectedModel.getLayerWiseConfigurations();

        MultiLayerNetwork modelToFineTune = new MultiLayerNetwork(confToChange,expectedModel.params());
        MultiLayerNetwork modelNow = new MLNTransferLearning
                                            //.Builder(modelToFineTune)
                                            (modelToFineTune)
                                            .fineTuneConfiguration(
                                                    new NeuralNetConfiguration.Builder()
                                                        .seed(rng)
                                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                                        .updater(Updater.RMSPROP)
                                                        .learningRate(0.0001)
                                                        .regularization(true))
                                            .build();

        assertTrue(expectedConf.toJson().equals(modelNow.getLayerWiseConfigurations().toJson()));

        modelNow.fit(randomData);
        expectedModel.fit(randomData);
        assertTrue(modelNow.params().equals(expectedModel.params()));
    }

    @Test
    public void testNoutChanges(){
        DataSet randomData = new DataSet(Nd4j.rand(10,4),Nd4j.rand(10,2));

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder().learningRate(0.1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD).activation(Activation.IDENTITY);
        MultiLayerNetwork modelToFineTune = new MultiLayerNetwork(overallConf.list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4).nOut(5)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(3).nOut(2)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(2).nOut(3)
                        .build())
                .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(3)
                        .build()).build());
        modelToFineTune.init();
        MultiLayerNetwork modelNow = new MLNTransferLearning(modelToFineTune)
                                        .fineTuneConfiguration(overallConf)
                                        .nOutReplace(3, 2, WeightInit.XAVIER, WeightInit.XAVIER)
                                        .nOutReplace(0, 3, WeightInit.XAVIER, WeightInit.XAVIER)
                                        .build();

        MultiLayerNetwork modelExpectedArch = new MultiLayerNetwork(overallConf.list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4).nOut(3)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(3).nOut(2)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(2).nOut(3)
                        .build())
                .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(2)
                        .build()).build());

        modelExpectedArch.init();

        //modelNow should have the same architecture as modelExpectedArch
        assertArrayEquals(modelExpectedArch.params().shape(), modelNow.params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(0).params().shape(), modelNow.getLayer(0).params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(1).params().shape(), modelNow.getLayer(1).params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(2).params().shape(), modelNow.getLayer(2).params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(3).params().shape(), modelNow.getLayer(3).params().shape());

        modelNow.setParams(modelExpectedArch.params());
        //fit should give the same results
        modelExpectedArch.fit(randomData);
        modelNow.fit(randomData);
        assertTrue(modelExpectedArch.params().equals(modelNow.params()));
    }


    @Test
    public void testPopandAdd() {
        DataSet randomData = new DataSet(Nd4j.rand(10,4),Nd4j.rand(10,3));

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder().learningRate(0.1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD).activation(Activation.IDENTITY);
        MultiLayerNetwork modelToFineTune = new MultiLayerNetwork(overallConf.list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4).nOut(5)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(5).nOut(2)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(2).nOut(3)
                        .build())
                .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(3)
                        .build()).build());
        modelToFineTune.init();
        MultiLayerNetwork modelNow = new MLNTransferLearning(modelToFineTune)
                .fineTuneConfiguration(overallConf)
                .nOutReplace(0, 7, WeightInit.XAVIER, WeightInit.XAVIER)
                .nOutReplace(2, 5, WeightInit.XAVIER)
                .popOutputLayer()
                .addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(5).nOut(3).activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork modelExpectedArch = new MultiLayerNetwork(overallConf.list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4).nOut(7)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(7).nOut(2)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(2).nOut(5)
                        .build())
                .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(5).nOut(3)
                        .build()).build());

        modelExpectedArch.init();

        //modelNow should have the same architecture as modelExpectedArch
        assertArrayEquals(modelExpectedArch.params().shape(), modelNow.params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(0).params().shape(), modelNow.getLayer(0).params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(1).params().shape(), modelNow.getLayer(1).params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(2).params().shape(), modelNow.getLayer(2).params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(3).params().shape(), modelNow.getLayer(3).params().shape());

        modelNow.setParams(modelExpectedArch.params());
        //fit should give the same results
        modelExpectedArch.fit(randomData);
        modelNow.fit(randomData);
        assertTrue(modelExpectedArch.params().equals(modelNow.params()));
    }


}
