package org.deeplearning4j.nn.layers.convolution;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;

/**
 * Created by susaneraly on 2/5/17.
 */
@Slf4j
public class FrozenLayerTest {

    /*
        A model with a few frozen layers ==
            Model with non frozen layers set with the output of the forward pass of the frozen layers
     */
    @Test
    public void testFrozen() {
        DataSet randomData = new DataSet(Nd4j.rand(10,4),Nd4j.rand(10,3));

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder().learningRate(0.1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD).activation(Activation.IDENTITY);
        MultiLayerNetwork modelToFineTune = new MultiLayerNetwork(overallConf.list()
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
                        .nIn(3).nOut(3)
                        .build()).build());

        modelToFineTune.init();
        INDArray asFrozenFeatures = modelToFineTune.feedForwardToLayer(2,randomData.getFeatures(),false).get(2);

        MultiLayerNetwork modelNow = new TransferLearning.Builder(modelToFineTune)
                .fineTuneConfiguration(overallConf)
                .setFeatureExtractor(1).build();

        MultiLayerNetwork notFrozen = new MultiLayerNetwork(overallConf.list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(2).nOut(3)
                        .build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(3)
                        .build()).build(),Nd4j.hstack(modelToFineTune.getLayer(2).params(),modelToFineTune.getLayer(3).params()));

        int i = 0;
        while (i<5) {
            notFrozen.fit(new DataSet(asFrozenFeatures,randomData.getLabels()));
            modelNow.fit(randomData);
            i++;
        }

        assertEquals(Nd4j.hstack(modelToFineTune.getLayer(0).params(),modelToFineTune.getLayer(1).params(),notFrozen.params()),modelNow.params());

    }


    @Test
    public void cloneMLNFrozen() {

        DataSet randomData = new DataSet(Nd4j.rand(10,4),Nd4j.rand(10,3));

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder().learningRate(0.1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD).activation(Activation.IDENTITY);
        MultiLayerNetwork modelToFineTune = new MultiLayerNetwork(overallConf.list()
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
                        .nIn(3).nOut(3)
                        .build()).build());

        modelToFineTune.init();
        INDArray asFrozenFeatures = modelToFineTune.feedForwardToLayer(2,randomData.getFeatures(),false).get(2);
        MultiLayerNetwork modelNow = new TransferLearning.Builder(modelToFineTune)
                .fineTuneConfiguration(overallConf)
                .setFeatureExtractor(1).build();

        MultiLayerNetwork clonedModel = modelNow.clone();

        //Check json
        assertEquals(clonedModel.getLayerWiseConfigurations().toJson(), modelNow.getLayerWiseConfigurations().toJson());

        //Check params
        assertEquals(modelNow.params(), clonedModel.params());

        MultiLayerNetwork notFrozen = new MultiLayerNetwork(overallConf.list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(2).nOut(3)
                        .build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(3)
                        .build()).build(),Nd4j.hstack(modelToFineTune.getLayer(2).params(),modelToFineTune.getLayer(3).params()));

        int i = 0;
        while (i<5) {
            notFrozen.fit(new DataSet(asFrozenFeatures,randomData.getLabels()));
            modelNow.fit(randomData);
            clonedModel.fit(randomData);
            i++;
        }

        INDArray expectedParams = Nd4j.hstack(modelToFineTune.getLayer(0).params(),modelToFineTune.getLayer(1).params(),notFrozen.params());
        assertEquals(expectedParams,modelNow.params());
        assertEquals(expectedParams,clonedModel.params());

    }
}
