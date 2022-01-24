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

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.eclipse.deeplearning4j.resources.DataSetResource;
import org.eclipse.deeplearning4j.resources.ResourceDataSets;
import org.nd4j.common.resources.Downloader;

import java.io.*;
import java.net.URL;

@Data
@NoArgsConstructor
@Slf4j
public class MnistFetcher {

    protected static final String LOCAL_DIR_NAME = "MNIST";

    protected File fileDir;

    private DataSetResource mnistTrain = ResourceDataSets.mnistTrain();
    private DataSetResource mnistTest = ResourceDataSets.mnistTest();
    private DataSetResource mnistTrainLabels = ResourceDataSets.mnistTrainLabels();
    private DataSetResource mnistTestLabels = ResourceDataSets.mnistTestLabels();


    public MnistFetcher(File tempDir) {
        this.fileDir = tempDir;
    }


    public String getName() {
        return "MNIST";
    }

    public File getBaseDir() {
        return DL4JResources.getDirectory(ResourceType.DATASET, getName());
    }



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


        mnistTrain.download(true,3,200000,20000);
        mnistTest.download(true,3,200000,20000);

        mnistTrainLabels.download(true,3,200000,20000);
        mnistTestLabels.download(true,3,200000,20000);

        // get labels
        fileDir = baseDir;
        return fileDir;
    }
}
