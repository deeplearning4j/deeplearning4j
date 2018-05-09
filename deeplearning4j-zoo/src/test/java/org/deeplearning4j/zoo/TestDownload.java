package org.deeplearning4j.zoo;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.zoo.model.LeNet;
import org.deeplearning4j.zoo.model.SimpleCNN;
import org.deeplearning4j.zoo.model.UNet;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
        if (ZooModel.ROOT_CACHE_DIR.exists())
            ZooModel.ROOT_CACHE_DIR.delete();

        // iterate through each available model
        ZooModel[] models = new ZooModel[]{
                LeNet.builder().numClasses(10).build(),
                SimpleCNN.builder().numClasses(10).build(),
                UNet.builder().numClasses(1).build()
        };


        for (int i = 0; i < models.length; i++) {
            log.info("Testing zoo model " + models[i].getClass().getName());
            ZooModel model = models[i];

            for (PretrainedType pretrainedType : PretrainedType.values()) {
                if (model.pretrainedAvailable(pretrainedType)) {
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
