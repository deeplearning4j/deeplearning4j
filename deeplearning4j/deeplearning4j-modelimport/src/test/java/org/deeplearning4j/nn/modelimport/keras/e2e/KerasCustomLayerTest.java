/*-
 *
 *  * Copyright 2017 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.nn.modelimport.keras.e2e;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.layers.custom.KerasLRN;
import org.deeplearning4j.nn.modelimport.keras.layers.custom.KerasPoolHelper;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

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

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    // run manually, might take a long time to load (too long for unit tests)
    @Ignore
    @Test
    public void testCustomLayerImport() throws Exception {
        // file paths
        String kerasWeightsAndConfigUrl = DL4JResources.getURLString("googlenet_keras_weightsandconfig.h5");
        File cachedKerasFile = testDir.newFile("googlenet_keras_weightsandconfig.h5");
        String outputPath = testDir.newFile("googlenet_dl4j_inference.zip").getAbsolutePath();

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
