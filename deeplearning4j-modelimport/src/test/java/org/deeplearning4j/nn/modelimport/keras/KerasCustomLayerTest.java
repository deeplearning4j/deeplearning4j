package org.deeplearning4j.nn.modelimport.keras;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.layers.custom.KerasLRN;
import org.deeplearning4j.nn.modelimport.keras.layers.custom.KerasPoolHelper;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;

import java.io.File;
import java.net.URL;

/**
 * Test import of Keras custom layers. Must be run manually, since user must download weights and config from
 * http://blob.deeplearning4j.org/models/googlenet_keras_weights.h5
 * http://blob.deeplearning4j.org/models/googlenet_config.json
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public class KerasCustomLayerTest {

    // run manually
    @Test
    public void testCustomLayerImport() throws Exception {
        // file paths
        String kerasWeightsAndConfigUrl = "http://blob.deeplearning4j.org/models/googlenet_keras_weightsandconfig.h5";
        File cachedKerasFile = new File(System.getProperty("java.io.tmpdir"), "googlenet_keras_weightsandconfig.h5");
        String outputPath = System.getProperty("java.io.tmpdir") + "/googlenet_dl4j_inference.zip";

        KerasLayer.registerCustomLayer("PoolHelper", KerasPoolHelper.class);
        KerasLayer.registerCustomLayer("LRN", KerasLRN.class);

        // download file
        if (!cachedKerasFile.exists()) {
            log.info("Downloading model to " + cachedKerasFile.toString());
            FileUtils.copyURLToFile(new URL(kerasWeightsAndConfigUrl), cachedKerasFile);
            cachedKerasFile.deleteOnExit();
        }

        org.deeplearning4j.nn.api.Model importedModel =
                        KerasModelImport.importKerasModelAndWeights(cachedKerasFile.getAbsolutePath());
        ModelSerializer.writeModel(importedModel, outputPath, false);

        ComputationGraph serializedModel = ModelSerializer.restoreComputationGraph(outputPath);
        log.info(serializedModel.summary());
    }
}
