package org.deeplearning4j.zoo;

import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.zoo.model.Darknet19;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.deeplearning4j.zoo.model.VGG19;
import org.deeplearning4j.zoo.model.YOLO2;
import org.deeplearning4j.zoo.util.ClassPrediction;
import org.deeplearning4j.zoo.util.Labels;
import org.deeplearning4j.zoo.util.darknet.COCOLabels;
import org.deeplearning4j.zoo.util.darknet.DarknetLabels;
import org.deeplearning4j.zoo.util.darknet.VOCLabels;
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Tests ImageNet utilities.
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public class TestImageNet extends BaseDL4JTest {

    @Test
    public void testImageNetLabels() throws IOException {
        // set up model
        ZooModel model = VGG19.builder().numClasses(0).build(); //num labels doesn't matter since we're getting pretrained imagenet
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

        // clean up for current model
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();
    }

    @Test
    public void testDarknetLabels() throws IOException {
        // set up model
        ZooModel model = Darknet19.builder().numClasses(0).build(); //num labels doesn't matter since we're getting pretrained imagenet
        ComputationGraph initializedModel = (ComputationGraph) model.initPretrained();

        // set up input and feedforward
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3, new ColorConversionTransform(COLOR_BGR2RGB));
        ClassLoader classloader = Thread.currentThread().getContextClassLoader();
        INDArray image = loader.asMatrix(classloader.getResourceAsStream("goldenretriever.jpg"));
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);
        INDArray result = initializedModel.outputSingle(image);
        Labels labels = new DarknetLabels();
        List<List<ClassPrediction>> predictions = labels.decodePredictions(result, 10);

        // check output labels of result
        log.info(predictions.toString());
        assertEquals("golden retriever", predictions.get(0).get(0).getLabel());

        // clean up for current model
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();

        // set up model
        model = TinyYOLO.builder().numClasses(0).build(); //num labels doesn't matter since we're getting pretrained imagenet
        initializedModel = (ComputationGraph) model.initPretrained();

        // set up input and feedforward
        loader = new NativeImageLoader(416, 416, 3, new ColorConversionTransform(COLOR_BGR2RGB));
        image = loader.asMatrix(classloader.getResourceAsStream("goldenretriever.jpg"));
        scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);
        INDArray outputs = initializedModel.outputSingle(image);
        List<DetectedObject> objs = YoloUtils.getPredictedObjects(Nd4j.create(((TinyYOLO) model).getPriorBoxes()), outputs, 0.6, 0.4);
        assertEquals(1, objs.size());

        // check output labels of result
        labels = new VOCLabels();
        for (DetectedObject obj : objs) {
            ClassPrediction classPrediction = labels.decodePredictions(obj.getClassPredictions(), 1).get(0).get(0);
            log.info(obj.toString() + " " + classPrediction);
            assertEquals("dog", classPrediction.getLabel());
        }

        // clean up for current model
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();

            // set up model
            model = YOLO2.builder().numClasses(1000).build(); //num labels doesn't matter since we're getting pretrained imagenet
            initializedModel = (ComputationGraph) model.initPretrained();

            // set up input and feedforward
            loader = new NativeImageLoader(608, 608, 3, new ColorConversionTransform(COLOR_BGR2RGB));
            image = loader.asMatrix(classloader.getResourceAsStream("goldenretriever.jpg"));
            scaler = new ImagePreProcessingScaler(0, 1);
            scaler.transform(image);
            outputs = initializedModel.outputSingle(image);
            objs = YoloUtils.getPredictedObjects(Nd4j.create(((YOLO2) model).getPriorBoxes()), outputs, 0.6, 0.4);
            assertEquals(1, objs.size());

        // check output labels of result
        labels = new COCOLabels();
        for (DetectedObject obj : objs) {
            ClassPrediction classPrediction = labels.decodePredictions(obj.getClassPredictions(), 1).get(0).get(0);
            log.info(obj.toString() + " " + classPrediction);
            assertEquals("dog", classPrediction.getLabel());
        }
    }

}
