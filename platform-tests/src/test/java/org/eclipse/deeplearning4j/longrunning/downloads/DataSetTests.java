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
package org.eclipse.deeplearning4j.longrunning.downloads;

import org.eclipse.deeplearning4j.resources.DataSetResource;
import org.eclipse.deeplearning4j.resources.ResourceDataSets;
import org.eclipse.deeplearning4j.resources.utils.EMnistSet;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.tests.tags.TagNames;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag(TagNames.DOWNLOADS)
public class DataSetTests {

    @Test
    public void testLfw(@TempDir Path tempDir) throws Exception {
        DataSetResource full = ResourceDataSets.lfwFullData(tempDir.toFile());
        downloadAndAssertExists(full);
        DataSetResource subset = ResourceDataSets.lfwSubData(tempDir.toFile());
        downloadAndAssertExists(subset);
    }


    @Test
    public void testCifar10(@TempDir Path tempDir) throws Exception {
        DataSetResource cifar10 = ResourceDataSets.cifar10(tempDir.toFile());
        downloadAndAssertExists(cifar10);
    }


    @Test
    public void testMnist(@TempDir Path tempDir) throws Exception {
        DataSetResource mnistTrain = ResourceDataSets.mnistTrain(tempDir.toFile());
        downloadAndAssertExists(mnistTrain);
        DataSetResource mnistTest = ResourceDataSets.mnistTest(tempDir.toFile());
        downloadAndAssertExists(mnistTest);
        DataSetResource mnistTestLabels = ResourceDataSets.mnistTestLabels(tempDir.toFile());
        downloadAndAssertExists(mnistTestLabels);
        DataSetResource mnistTrainLabels = ResourceDataSets.mnistTrainLabels(tempDir.toFile());
        downloadAndAssertExists(mnistTrainLabels);

    }


    @Test
    public void testEMnist(@TempDir Path tempDir) throws Exception {
        for(EMnistSet set : EMnistSet.values()) {
            DataSetResource emnistDataTrain = ResourceDataSets.emnistTrain(set,tempDir.toFile());
            downloadAndAssertExists(emnistDataTrain);
            DataSetResource emnistDataTest = ResourceDataSets.emnistTest(set,tempDir.toFile());
            downloadAndAssertExists(emnistDataTest);
            DataSetResource emnistLabelsTrain = ResourceDataSets.emnistLabelsTrain(set,tempDir.toFile());
            downloadAndAssertExists(emnistLabelsTrain);
            DataSetResource emnistLabelsTest = ResourceDataSets.emnistLabelsTest(set,tempDir.toFile());
            downloadAndAssertExists(emnistLabelsTest);
            DataSetResource emnistMappingTrain = ResourceDataSets.emnistMappingTrain(set,tempDir.toFile());
            downloadAndAssertExists(emnistMappingTrain,false);
            DataSetResource emnistMappingTest = ResourceDataSets.emnistMappingTest(set,tempDir.toFile());
            downloadAndAssertExists(emnistMappingTest,false);
        }
    }


    private void downloadAndAssertExists(DataSetResource resource,boolean archive) throws Exception {
        resource.download(archive,3,3000,3000);
        assertTrue(resource.existsLocally());
        resource.delete();
        assertFalse(resource.existsLocally());
    }

    private void downloadAndAssertExists(DataSetResource resource) throws Exception {
        downloadAndAssertExists(resource,true);
    }


}
