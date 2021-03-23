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

package org.deeplearning4j.zoo;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.zoo.model.LeNet;
import org.deeplearning4j.zoo.model.NASNet;
import org.deeplearning4j.zoo.model.SimpleCNN;
import org.deeplearning4j.zoo.model.UNet;
import org.deeplearning4j.zoo.util.darknet.COCOLabels;
import org.deeplearning4j.zoo.util.darknet.DarknetLabels;
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

@Slf4j
@Tag(TagNames.FILE_IO)
@Tag(TagNames.DL4J_OLD_API)
@NativeTag
@Tag(TagNames.LONG_TEST)
public class TestDownload extends BaseDL4JTest {
    @TempDir
    static Path sharedTempDir;

    @Override
    public long getTimeoutMilliseconds() {
        return isIntegrationTests() ? 480000L : 60000L;
    }



    @BeforeAll
    public static void before() throws Exception {
        DL4JResources.setBaseDirectory(sharedTempDir.toFile());
    }

    @AfterAll
    public static void after(){
        DL4JResources.resetBaseDirectoryLocation();
    }

    @Test
    public void testDownloadAllModels() throws Exception {

        // iterate through each available model
        ZooModel[] models;

        if(isIntegrationTests()){
            models = new ZooModel[]{
                    LeNet.builder().build(),
                    SimpleCNN.builder().build(),
                    UNet.builder().build(),
                    NASNet.builder().build()};
        } else {
            models = new ZooModel[]{
                    LeNet.builder().build(),
                    SimpleCNN.builder().build()};
        }



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


    @Test
    public void testLabelsDownload() throws Exception {
        assertEquals("person", new COCOLabels().getLabel(0));
        assertEquals("kit fox", new DarknetLabels(true).getLabel(0));
        assertEquals("n02119789", new DarknetLabels(false).getLabel(0));
        assertEquals("tench", new ImageNetLabels().getLabel(0));
    }
}
