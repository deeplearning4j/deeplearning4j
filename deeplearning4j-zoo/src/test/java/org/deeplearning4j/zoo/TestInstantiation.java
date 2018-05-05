package org.deeplearning4j.zoo;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.BenchmarkDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.zoo.model.*;
import org.deeplearning4j.zoo.model.helper.DarknetHelper;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.Map;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertArrayEquals;

/**
 * Tests workflow for zoo model instantiation.
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public class TestInstantiation {

    @Test
    public void testMultipleCnnTraining() throws Exception {
        int numClasses = 10;
        ZooModel[] models = new ZooModel[]{
                Darknet19.builder().numClasses(numClasses).build(),
                TinyYOLO.builder().numClasses(numClasses).build(),
                YOLO2.builder().numClasses(numClasses).build()
        };

        for(int i = 0; i < models.length; i++) {
            ZooModel model = models[i];
            String modelName = model.getClass().getName();
            log.info("Testing training on zoo model " + modelName);
            int gridWidth = -1;
            int gridHeight = -1;
            if (modelName.equals("TinyYOLO") || modelName.equals("YOLO2")) {
                int[] inputShapes = model.metaData().getInputShape()[0];
                gridWidth = DarknetHelper.getGridWidth(inputShapes);
                gridHeight = DarknetHelper.getGridHeight(inputShapes);
                numClasses += 4;
            }

            // set up data iterator
            int[] inputShape = model.metaData().getInputShape()[0];
            DataSetIterator iter = new BenchmarkDataSetIterator(
                            new int[] {16, inputShape[0], inputShape[1], inputShape[2]}, numClasses, 1,
                            gridWidth, gridHeight);

            Model initializedModel = model.init();
            while (iter.hasNext()) {
                DataSet ds = iter.next();
                if (initializedModel instanceof ComputationGraph)
                    ((ComputationGraph) initializedModel).fit(ds);
                else if (initializedModel instanceof MultiLayerNetwork)
                    ((MultiLayerNetwork) initializedModel).fit(ds);
                else
                    throw new IllegalStateException("Zoo models are only MultiLayerNetwork or ComputationGraph.");
            }

            // clean up for current model
            Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
            System.gc();
            Thread.sleep(1000);
        }
    }

    @Test
    public void testInitPretrained() throws IOException {
        ZooModel model = ResNet50.builder().numClasses(0).build(); //num labels doesn't matter since we're getting pretrained imagenet
        assertTrue(model.pretrainedAvailable(PretrainedType.IMAGENET));

        ComputationGraph initializedModel = (ComputationGraph) model.initPretrained();
        INDArray[] result = initializedModel.output(Nd4j.rand(new int[] {1, 3, 224, 224}));
        assertArrayEquals(result[0].shape(), new int[] {1, 1000});

        // clean up for current model
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();

        model = VGG16.builder().numClasses(0).build(); //num labels doesn't matter since we're getting pretrained imagenet
        assertTrue(model.pretrainedAvailable(PretrainedType.IMAGENET));

        initializedModel = (ComputationGraph) model.initPretrained();
        result = initializedModel.output(Nd4j.rand(new int[] {1, 3, 224, 224}));
        assertArrayEquals(result[0].shape(), new int[] {1, 1000});

        // clean up for current model
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();

        model = Darknet19.builder().numClasses(0).build(); //num labels doesn't matter since we're getting pretrained imagenet
        assertTrue(model.pretrainedAvailable(PretrainedType.IMAGENET));

        initializedModel = (ComputationGraph) model.initPretrained();
        result = initializedModel.output(Nd4j.rand(new int[] {1, 3, 224, 224}));
        assertArrayEquals(result[0].shape(), new int[] {1, 1000});

        // clean up for current model
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();

        model = Darknet19.builder().numClasses(0).build(); //num labels doesn't matter since we're getting pretrained imagenet
        model.setInputShape(new int[][] {{3, 448, 448}});
        assertTrue(model.pretrainedAvailable(PretrainedType.IMAGENET));

        initializedModel = (ComputationGraph) model.initPretrained();
        result = initializedModel.output(Nd4j.rand(new int[] {1, 3, 448, 448}));
        assertArrayEquals(result[0].shape(), new int[] {1, 1000});

        // clean up for current model
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();

        model = TinyYOLO.builder().numClasses(0).build(); //num labels doesn't matter since we're getting pretrained imagenet
        assertTrue(model.pretrainedAvailable(PretrainedType.IMAGENET));

        initializedModel = (ComputationGraph) model.initPretrained();
        result = initializedModel.output(Nd4j.rand(new int[] {1, 3, 416, 416}));
        assertArrayEquals(result[0].shape(), new int[] {1, 125, 13, 13});

        // clean up for current model
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();

        model = YOLO2.builder().numClasses(0).build(); //num labels doesn't matter since we're getting pretrained imagenet
        assertTrue(model.pretrainedAvailable(PretrainedType.IMAGENET));

        initializedModel = (ComputationGraph) model.initPretrained();
        result = initializedModel.output(Nd4j.rand(new int[] {1, 3, 608, 608}));
        assertArrayEquals(result[0].shape(), new int[] {1, 425, 19, 19});

        // clean up for current model
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();

        model = Xception.builder().numClasses(0).build(); //num labels doesn't matter since we're getting pretrained imagenet
        assertTrue(model.pretrainedAvailable(PretrainedType.IMAGENET));

        initializedModel = (ComputationGraph) model.initPretrained();
        result = initializedModel.output(Nd4j.rand(new int[] {1, 3, 299, 299}));
        assertArrayEquals(result[0].shape(), new int[] {1, 1000});

        // clean up for current model
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();

        model = SqueezeNet.builder().numClasses(0).build(); //num labels doesn't matter since we're getting pretrained imagenet
        assertTrue(model.pretrainedAvailable(PretrainedType.IMAGENET));

        initializedModel = (ComputationGraph) model.initPretrained();
        log.info(initializedModel.summary());
        result = initializedModel.output(Nd4j.rand(new int[] {1, 3, 227, 227}));
        assertArrayEquals(result[0].shape(), new int[] {1, 1000, 1, 1});
    }


    @Test
    public void testYolo4635() throws Exception {
        //https://github.com/deeplearning4j/deeplearning4j/issues/4635

        int nClasses = 10;
        TinyYOLO model = TinyYOLO.builder().numClasses(nClasses).build();
        ComputationGraph computationGraph = (ComputationGraph) model.initPretrained();
        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(computationGraph, "conv2d_9");
    }

}
