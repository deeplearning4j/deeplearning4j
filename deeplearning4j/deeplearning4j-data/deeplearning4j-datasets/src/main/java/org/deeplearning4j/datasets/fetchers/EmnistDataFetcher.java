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

package org.deeplearning4j.datasets.fetchers;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.base.EmnistFetcher;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.eclipse.deeplearning4j.resources.utils.EMnistSet;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.MnistManager;
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher;

import java.io.File;
import java.io.IOException;
import java.util.Random;


@Slf4j
public class EmnistDataFetcher extends MnistDataFetcher implements DataSetFetcher {

    protected EmnistFetcher fetcher;



    public EmnistDataFetcher(EMnistSet dataSet, boolean binarize, boolean train, boolean shuffle,
                             long rngSeed,File topLevelDir) throws IOException {
        fetcher = new EmnistFetcher(dataSet,topLevelDir);
        if (!emnistExists(fetcher)) {
            fetcher.downloadAndUntar();
        }


        String EMNIST_ROOT = topLevelDir.getAbsolutePath();
        if (train) {
            images = fetcher.getEmnistDataTrain().localPath().getAbsolutePath();
            labels = fetcher.getEmnistLabelsTrain().localPath().getAbsolutePath();
            totalExamples = EmnistDataSetIterator.numExamplesTrain(dataSet);
        } else {
            images = fetcher.getEmnistDataTest().localPath().getAbsolutePath();
            labels = fetcher.getEmnistLabelsTest().localPath().getAbsolutePath();
            totalExamples = EmnistDataSetIterator.numExamplesTest(dataSet);
        }
        try {
            manager = new MnistManager(images, labels, totalExamples);
        } catch (Exception e) {
            log.error("",e);
            FileUtils.deleteDirectory(new File(EMNIST_ROOT));
            new EmnistFetcher(dataSet).downloadAndUntar();
            manager = new MnistManager(images, labels, totalExamples);
        }

        numOutcomes = EmnistDataSetIterator.numLabels(dataSet);
        this.binarize = binarize;
        cursor = 0;
        manager.setCurrent(cursor);
        inputColumns = manager.getImages().getEntryLength();
        this.train = train;
        this.shuffle = shuffle;

        order = new int[totalExamples];
        for (int i = 0; i < order.length; i++)
            order[i] = i;
        rng = new Random(rngSeed);
        reset(); //Shuffle order


        //For some inexplicable reason, EMNIST LETTERS set is indexed 1 to 26 (i.e., 1 to nClasses), while everything else
        // is indexed (0 to nClasses-1) :/
        if (dataSet == EMnistSet.LETTERS) {
            oneIndexed = true;
        } else {
            oneIndexed = false;
        }
        this.fOrder = true; //MNIST is C order, EMNIST is F order
    }

    public EmnistDataFetcher(EMnistSet dataSet, boolean binarize, boolean train, boolean shuffle,
                             long rngSeed) throws IOException {
        this(dataSet,binarize,train,shuffle,rngSeed,DL4JResources.getDirectory(ResourceType.DATASET, "EMNIST"));
    }

    private boolean emnistExists(EmnistFetcher e) {
        //Check 4 files:
        if (!fetcher.getEmnistDataTrain().existsLocally())
            return false;
        if (!fetcher.getEmnistLabelsTrain().existsLocally())
            return false;
        if (!fetcher.getEmnistDataTest().existsLocally())
            return false;
        if (!fetcher.getEmnistLabelsTest().existsLocally())
            return false;
        return true;
    }
}
