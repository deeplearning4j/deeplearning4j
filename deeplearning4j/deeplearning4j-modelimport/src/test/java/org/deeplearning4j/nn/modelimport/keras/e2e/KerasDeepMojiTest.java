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
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.KerasSequentialModel;
import org.deeplearning4j.nn.modelimport.keras.layers.custom.KerasDeepMojiAttention;
import org.deeplearning4j.nn.modelimport.keras.layers.custom.KerasLRN;
import org.deeplearning4j.nn.modelimport.keras.layers.custom.KerasPoolHelper;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelBuilder;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

import static java.io.File.createTempFile;

/**
 * Test import of DeepMoji application
 *
 * @author Max Pumperla
 */
@Slf4j
public class KerasDeepMojiTest {

    private static final String TEMP_MODEL_FILENAME = "tempModel";
    private static final String H5_EXTENSION = ".h5";

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    // run manually, might take a long time to load (too long for unit tests)
    //@Ignore
    @Test
    public void testDeepMojiImport() throws Exception {

        KerasLayer.registerCustomLayer("AttentionWeightedAverage", KerasDeepMojiAttention.class);
        importFunctionalModelH5Test("modelimport/keras/examples/DeepMoji/deepmoji.h5", null, false);
    }

    private ComputationGraph importFunctionalModelH5Test(String modelPath, int[] inputShape, boolean train) throws Exception {
        ClassPathResource modelResource =
                new ClassPathResource(modelPath,
                        KerasModelEndToEndTest.class.getClassLoader());
        File modelFile = createTempFile(TEMP_MODEL_FILENAME, H5_EXTENSION);
        Files.copy(modelResource.getInputStream(), modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        KerasModelBuilder builder = new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath())
                .enforceTrainingConfig(train);
        if (inputShape != null) {
            builder.inputShape(inputShape);
        }
        KerasModel model = builder.buildModel();
        return model.getComputationGraph();
    }
}
