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

package org.deeplearning4j.datasets.base;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.resources.DataSetResource;
import org.eclipse.deeplearning4j.resources.ResourceDataSets;
import org.eclipse.deeplearning4j.resources.utils.EMnistSet;

import java.io.File;
import java.io.IOException;

@Slf4j
public class EmnistFetcher extends MnistFetcher {

    private final EMnistSet ds;
    @Getter
    private DataSetResource emnistDataTrain;
    @Getter
    private DataSetResource emnistDataTest;
    @Getter
    private DataSetResource emnistLabelsTrain;
    @Getter
    private DataSetResource emnistLabelsTest;
    @Getter
    private DataSetResource emnistMappingTrain;
    @Getter
    private DataSetResource emnistMappingTest;

    public EmnistFetcher() {
        this(EMnistSet.MNIST);
    }

    public EmnistFetcher(EMnistSet ds) {
        this.ds = ds;
        emnistDataTrain = ResourceDataSets.emnistTrain(ds);
        emnistDataTest = ResourceDataSets.emnistTest(ds);
        emnistLabelsTrain = ResourceDataSets.emnistLabelsTrain(ds);
        emnistLabelsTest = ResourceDataSets.emnistLabelsTest(ds);
        emnistMappingTrain = ResourceDataSets.emnistMappingTrain(ds);
        emnistMappingTest = ResourceDataSets.emnistMappingTest(ds);

    }

    public EmnistFetcher(EMnistSet ds,File topLevelDir) {
        this.ds = ds;
        emnistDataTrain = ResourceDataSets.emnistTrain(ds,topLevelDir);
        emnistDataTest = ResourceDataSets.emnistTest(ds,topLevelDir);
        emnistLabelsTrain = ResourceDataSets.emnistLabelsTrain(ds,topLevelDir);
        emnistLabelsTest = ResourceDataSets.emnistLabelsTest(ds,topLevelDir);
        emnistMappingTrain = ResourceDataSets.emnistMappingTrain(ds,topLevelDir);
        emnistMappingTest = ResourceDataSets.emnistMappingTest(ds,topLevelDir);

    }


    @Override
    public String getName() {
        return "EMNIST";
    }

    // --- Train files ---

    public static int numLabels(EMnistSet dataSet) {
        switch (dataSet) {
            case COMPLETE:
                return 62;
            case MERGE:
                return 47;
            case BALANCED:
                return 47;
            case LETTERS:
                return 26;
            case DIGITS:
                return 10;
            case MNIST:
                return 10;
            default:
                throw new UnsupportedOperationException("Unknown Set: " + dataSet);
        }
    }

    @Override
    public File downloadAndUntar() throws IOException {
        if (fileDir != null) {
            return fileDir;
        }

        File baseDir = getBaseDir();
        if (!(baseDir.isDirectory() || baseDir.mkdir())) {
            throw new IOException("Could not mkdir " + baseDir);
        }

        log.info("Downloading {}...", getName());
        // get features
        emnistDataTrain.download(true,3,300000,30000);
        emnistDataTest.download(true,3,300000,30000);
        emnistLabelsTrain.download(true,3,300000,30000);
        emnistLabelsTest.download(true,3,300000,30000);
        emnistMappingTrain.download(false,3,300000,30000);
        emnistMappingTest.download(false,3,300000,30000);


        // get labels
        fileDir = baseDir;
        return fileDir;
    }
}
