package org.deeplearning4j.nn.conf;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.*;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 26/06/2017.
 */
public class JsonTest {

    @Test
    public void testJsonLossFunctions() {

        ILossFunction[] lossFunctions = new ILossFunction[] {new LossBinaryXENT(), new LossBinaryXENT(),
                        new LossCosineProximity(), new LossHinge(), new LossKLD(), new LossKLD(), new LossL1(),
                        new LossL1(), new LossL2(), new LossL2(), new LossMAE(), new LossMAE(), new LossMAPE(),
                        new LossMAPE(), new LossMCXENT(), new LossMSE(), new LossMSE(), new LossMSLE(), new LossMSLE(),
                        new LossNegativeLogLikelihood(), new LossNegativeLogLikelihood(), new LossPoisson(),
                        new LossSquaredHinge(), new LossFMeasure(), new LossFMeasure(2.0)};

        String[] outputActivationFn = new String[] {"sigmoid", //xent
                        "sigmoid", //xent
                        "tanh", //cosine
                        "tanh", //hinge -> trying to predict 1 or -1
                        "sigmoid", //kld -> probab so should be between 0 and 1
                        "softmax", //kld + softmax
                        "tanh", //l1
                        "softmax", //l1 + softmax
                        "tanh", //l2
                        "softmax", //l2 + softmax
                        "identity", //mae
                        "softmax", //mae + softmax
                        "identity", //mape
                        "softmax", //mape + softmax
                        "softmax", //mcxent
                        "identity", //mse
                        "softmax", //mse + softmax
                        "sigmoid", //msle  -   requires positive labels/activations due to log
                        "softmax", //msle + softmax
                        "sigmoid", //nll
                        "softmax", //nll + softmax
                        "sigmoid", //poisson - requires positive predictions due to log... not sure if this is the best option
                        "tanh", //squared hinge
                        "sigmoid", //f-measure (binary, single sigmoid output)
                        "softmax" //f-measure (binary, 2-label softmax output)
        };

        int[] nOut = new int[] {1, //xent
                        3, //xent
                        5, //cosine
                        3, //hinge
                        3, //kld
                        3, //kld + softmax
                        3, //l1
                        3, //l1 + softmax
                        3, //l2
                        3, //l2 + softmax
                        3, //mae
                        3, //mae + softmax
                        3, //mape
                        3, //mape + softmax
                        3, //mcxent
                        3, //mse
                        3, //mse + softmax
                        3, //msle
                        3, //msle + softmax
                        3, //nll
                        3, //nll + softmax
                        3, //poisson
                        3, //squared hinge
                        1, //f-measure (binary, single sigmoid output)
                        2, //f-measure (binary, 2-label softmax output)
        };

        for (int i = 0; i < lossFunctions.length; i++) {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).updater(Updater.ADAM).list()
                            .layer(0, new DenseLayer.Builder().nIn(4).nOut(nOut[i]).activation(Activation.TANH).build())
                            .layer(1, new LossLayer.Builder().lossFunction(lossFunctions[i])
                                            .activation(outputActivationFn[i]).build())
                            .pretrain(false).backprop(true).build();

            String json = conf.toJson();
            String yaml = conf.toYaml();

            MultiLayerConfiguration fromJson = MultiLayerConfiguration.fromJson(json);
            MultiLayerConfiguration fromYaml = MultiLayerConfiguration.fromYaml(yaml);

            assertEquals(conf, fromJson);
            assertEquals(conf, fromYaml);
        }
    }

}
