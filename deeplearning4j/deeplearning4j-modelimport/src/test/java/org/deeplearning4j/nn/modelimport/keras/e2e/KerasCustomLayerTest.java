/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.nn.modelimport.keras.e2e;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.layers.custom.KerasLRN;
import org.deeplearning4j.nn.modelimport.keras.layers.custom.KerasPoolHelper;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.jupiter.api.Disabled;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.io.File;
import java.net.URL;
import org.junit.jupiter.api.DisplayName;
import java.nio.file.Path;
import org.junit.jupiter.api.extension.ExtendWith;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

@Slf4j
@DisplayName("Keras Custom Layer Test")
@Tag(TagNames.FILE_IO)
@Tag(TagNames.KERAS)
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
class KerasCustomLayerTest extends BaseDL4JTest {

    @TempDir
    public Path testDir;

    // run manually, might take a long time to load (too long for unit tests)
    @Disabled
    @Test
    @DisplayName("Test Custom Layer Import")
    void testCustomLayerImport() throws Exception {
        // file paths
        String kerasWeightsAndConfigUrl = DL4JResources.getURLString("googlenet_keras_weightsandconfig.h5");
        File cachedKerasFile = testDir.resolve("googlenet_keras_weightsandconfig.h5").toFile();
        File newFile = new File(testDir.toFile(),"googlenet_dl4j_inference.zip");
        String outputPath = newFile.getAbsolutePath();
        KerasLayer.registerCustomLayer("PoolHelper", KerasPoolHelper.class);
        KerasLayer.registerCustomLayer("LRN", KerasLRN.class);
        // download file
        if (!cachedKerasFile.exists()) {
            log.info("Downloading model to " + cachedKerasFile.toString());
            FileUtils.copyURLToFile(new URL(kerasWeightsAndConfigUrl), cachedKerasFile);
            cachedKerasFile.deleteOnExit();
        }
        org.deeplearning4j.nn.api.Model importedModel = KerasModelImport.importKerasModelAndWeights(cachedKerasFile.getAbsolutePath());
        ModelSerializer.writeModel(importedModel, outputPath, false);
        ComputationGraph serializedModel = ModelSerializer.restoreComputationGraph(outputPath);
        log.info(serializedModel.summary());
    }
}
