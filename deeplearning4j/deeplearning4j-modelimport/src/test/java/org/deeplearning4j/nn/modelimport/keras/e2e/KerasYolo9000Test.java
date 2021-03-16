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
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasSpaceToDepth;
import org.junit.jupiter.api.Disabled;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.resources.Resources;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import org.junit.jupiter.api.DisplayName;
import java.nio.file.Path;
import org.junit.jupiter.api.extension.ExtendWith;

@Slf4j
@DisplayName("Keras Yolo 9000 Test")
class KerasYolo9000Test extends BaseDL4JTest {

    private static final String TEMP_MODEL_FILENAME = "tempModel";

    private static final String H5_EXTENSION = ".h5";

    @TempDir
    public Path testDir;

    @Disabled
    @Test
    @DisplayName("Test Custom Layer Yolo Import")
    // TODO: yolo and yolo-voc output are too large for github, find smaller equivalents
    void testCustomLayerYoloImport() throws Exception {
        KerasLayer.registerCustomLayer("Lambda", KerasSpaceToDepth.class);
        String modelPath = "modelimport/keras/examples/yolo/yolo.h5";
        try (InputStream is = Resources.asStream(modelPath)) {
            File modelFile = testDir.resolve(TEMP_MODEL_FILENAME + System.currentTimeMillis() + H5_EXTENSION).toFile();
            Files.copy(is, modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            ComputationGraph model = new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath()).enforceTrainingConfig(false).buildModel().getComputationGraph();
            System.out.println(model.summary());
        }
    }
}
