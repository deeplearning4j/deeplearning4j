package org.deeplearning4j.nn.layers.capsule;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CapsuleLayer;
import org.deeplearning4j.nn.conf.layers.CapsuleStrengthLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.PrimaryCapsules;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.learning.config.Adam;

public class CapsNetMNISTTest extends BaseDL4JTest {
    @Test
    public void testCapsNetOnMNIST(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam())
                .list()
                .layer(new ConvolutionLayer.Builder()
                        .nOut(16)
                        .kernelSize(9, 9)
                        .stride(3, 3)
                        .build())
                .layer(new PrimaryCapsules.Builder(8, 8)
                        .kernelSize(7, 7)
                        .stride(2, 2)
                        .build())
                .layer(new CapsuleLayer.Builder(10, 16, 3).build())
                .layer(new CapsuleStrengthLayer.Builder().build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        int rngSeed = 12345;
        try {
            MnistDataSetIterator mnistTrain = new MnistDataSetIterator(64, true, rngSeed);
            MnistDataSetIterator mnistTest = new MnistDataSetIterator(64, false, rngSeed);

            for (int i = 0; i < 2; i++) {
                model.fit(mnistTrain);
            }

            Evaluation eval = model.evaluate(mnistTest);

            assertTrue("Accuracy not over 95%", eval.accuracy() > 0.95);
            assertTrue("Precision not over 95%", eval.precision() > 0.95);
            assertTrue("Recall not over 95%", eval.recall() > 0.95);
            assertTrue("F1-score not over 95%", eval.f1() > 0.95);

        } catch (IOException e){
            System.out.println("Could not load MNIST.");
        }
    }
}
