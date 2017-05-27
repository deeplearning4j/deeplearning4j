package org.deeplearning4j.zoo;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.BenchmarkDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.zoo.model.GoogLeNet;
import org.deeplearning4j.zoo.model.ResNet50;
import org.deeplearning4j.zoo.model.VGG16;
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
        Map<ZooType, ZooModel> models = ModelSelector.select(ZooType.VGG19, 10);

        for (Map.Entry<ZooType, ZooModel> entry : models.entrySet()) {
            log.info("Testing training on zoo model " + entry.getKey());
            ZooModel model = entry.getValue();

            // set up data iterator
            int[] inputShape = model.metaData().getInputShape()[0];
            DataSetIterator iter = new BenchmarkDataSetIterator(
                            new int[] {16, inputShape[0], inputShape[1], inputShape[2]}, 10, 1);

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
        ZooModel model = new ResNet50(); //num labels doesn't matter since we're getting pretrained imagenet
        assertTrue(model.pretrainedAvailable(PretrainedType.IMAGENET));

        ComputationGraph initializedModel = (ComputationGraph) model.initPretrained();
        INDArray[] result = initializedModel.output(Nd4j.rand(new int[] {1, 3, 224, 224}));
        assertArrayEquals(result[0].shape(), new int[] {1, 1000});

        // clean up for current model
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();

        model = new VGG16(); //num labels doesn't matter since we're getting pretrained imagenet
        assertTrue(model.pretrainedAvailable(PretrainedType.IMAGENET));

        initializedModel = (ComputationGraph) model.initPretrained();
        result = initializedModel.output(Nd4j.rand(new int[] {1, 3, 224, 224}));
        assertArrayEquals(result[0].shape(), new int[] {1, 1000});

        // clean up for current model
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();

        model = new GoogLeNet(); //num labels doesn't matter since we're getting pretrained imagenet
        assertTrue(model.pretrainedAvailable(PretrainedType.IMAGENET));

        initializedModel = (ComputationGraph) model.initPretrained();
        result = initializedModel.output(Nd4j.rand(new int[] {1, 3, 224, 224}));
        assertArrayEquals(result[0].shape(), new int[] {1, 1000});
    }

}
