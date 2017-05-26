package org.deeplearning4j.zoo;

import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.model.GoogLeNet;
import org.deeplearning4j.zoo.model.ResNet50;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.model.VGG19;
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import java.io.IOException;

import static org.junit.Assert.assertTrue;

/**
 * Tests ImageNet utilities.
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public class TestImageNet {

    @Test
    public void testImageNetLabels() throws IOException {
        // set up model
        ZooModel model = new VGG19(1, 123, 1); //num labels doesn't matter since we're getting pretrained imagenet
        ComputationGraph initializedModel = (ComputationGraph) model.initPretrained();

        // set up input and feedforward
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        ClassLoader classloader = Thread.currentThread().getContextClassLoader();
        INDArray image = loader.asMatrix(classloader.getResourceAsStream("goldenretriever.jpg"));
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);
        INDArray[] output = initializedModel.output(false, image);

        // check output labels of result
        String decodedLabels = new ImageNetLabels().decodePredictions(output[0]);
        log.info(decodedLabels);
        assertTrue(decodedLabels.contains("golden_retriever"));
    }

}
