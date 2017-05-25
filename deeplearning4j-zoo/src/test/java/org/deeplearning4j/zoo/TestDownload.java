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
 * Tests downloads and checksum verification.
 *
 * @note This test first deletes the ~/.deeplearning4j/ cache directory.
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public class TestDownload {

    @Test
    public void testDownloadAllModels() throws Exception {
        // clean up
        if(ZooModel.ROOT_CACHE_DIR.exists())
            ZooModel.ROOT_CACHE_DIR.delete();

        // iterate through each available model
        Map<ZooType, ZooModel> models = ModelSelector.select(ZooType.CNN, 10);

        for (Map.Entry<ZooType, ZooModel> entry : models.entrySet()) {
            log.info("Testing zoo model " + entry.getKey());
            ZooModel model = entry.getValue();

            for(PretrainedType pretrainedType : PretrainedType.values()) {
                if(model.pretrainedAvailable(pretrainedType)) {
                    model.initPretrained(pretrainedType);
                }
            }

            // clean up for current model
            Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
            System.gc();
            Thread.sleep(1000);
        }
    }

}
