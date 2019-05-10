/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.modelimport.keras.e2e;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasSpaceToDepth;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

/**
 * Import previously stored YOLO9000 Keras net from https://github.com/allanzelener/YAD2K.
 * <p>
 * git clone https://github.com/allanzelener/YAD2K
 * cd YAD2K
 * wget http://pjreddie.com/media/files/yolo.weights
 * wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg
 * python3 yad2k.py yolo.cfg yolo.weights yolo.h5
 * <p>
 * To run this test put the output of this script on the test resources path.
 *
 * @author Max Pumperla
 */
@Slf4j
public class KerasYolo9000Test {

    private static final String TEMP_MODEL_FILENAME = "tempModel";
    private static final String H5_EXTENSION = ".h5";

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Ignore
    @Test
    // TODO: yolo and yolo-voc output are too large for github, find smaller equivalents
    public void testCustomLayerYoloImport() throws Exception {
        KerasLayer.registerCustomLayer("Lambda", KerasSpaceToDepth.class);

        String modelPath = "modelimport/keras/examples/yolo/yolo.h5";

        ClassPathResource modelResource =
                new ClassPathResource(modelPath,
                        KerasModelEndToEndTest.class.getClassLoader());
        File modelFile = testDir.newFile(TEMP_MODEL_FILENAME + System.currentTimeMillis() + H5_EXTENSION);
        Files.copy(modelResource.getInputStream(), modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        ComputationGraph model = new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath())
                .enforceTrainingConfig(false).buildModel().getComputationGraph();

        System.out.println(model.summary());


    }
}
