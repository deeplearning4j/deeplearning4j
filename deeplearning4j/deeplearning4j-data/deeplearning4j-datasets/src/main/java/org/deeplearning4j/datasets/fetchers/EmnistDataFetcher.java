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
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.MnistManager;
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher;

import java.io.File;
import java.io.IOException;
import java.util.Random;


@Slf4j
public class EmnistDataFetcher extends MnistDataFetcher implements DataSetFetcher {

    protected EmnistFetcher fetcher;

    public EmnistDataFetcher(EmnistDataSetIterator.Set dataSet, boolean binarize, boolean train, boolean shuffle,
                             long rngSeed) throws IOException {
        fetcher = new EmnistFetcher(dataSet);
        if (!emnistExists(fetcher)) {
            fetcher.downloadAndUntar();
        }


        String EMNIST_ROOT = DL4JResources.getDirectory(ResourceType.DATASET, "EMNIST").getAbsolutePath();
        if (train) {
            images = FilenameUtils.concat(EMNIST_ROOT, fetcher.getTrainingFilesFilename_unzipped());
            labels = FilenameUtils.concat(EMNIST_ROOT, fetcher.getTrainingFileLabelsFilename_unzipped());
            totalExamples = EmnistDataSetIterator.numExamplesTrain(dataSet);
        } else {
            images = FilenameUtils.concat(EMNIST_ROOT, fetcher.getTestFilesFilename_unzipped());
            labels = FilenameUtils.concat(EMNIST_ROOT, fetcher.getTestFileLabelsFilename_unzipped());
            totalExamples = EmnistDataSetIterator.numExamplesTest(dataSet);
        }
        MnistManager man;
        try {
            man = new MnistManager(images, labels, totalExamples);
        } catch (Exception e) {
            log.error("",e);
            FileUtils.deleteDirectory(new File(EMNIST_ROOT));
            new EmnistFetcher(dataSet).downloadAndUntar();
            man = new MnistManager(images, labels, totalExamples);
        }

        numOutcomes = EmnistDataSetIterator.numLabels(dataSet);
        this.binarize = binarize;
        cursor = 0;
        man.setCurrent(cursor);
        inputColumns = man.getImages().getEntryLength();
        this.train = train;
        this.shuffle = shuffle;

        order = new int[totalExamples];
        for (int i = 0; i < order.length; i++)
            order[i] = i;
        rng = new Random(rngSeed);
        reset(); //Shuffle order


        //For some inexplicable reason, EMNIST LETTERS set is indexed 1 to 26 (i.e., 1 to nClasses), while everything else
        // is indexed (0 to nClasses-1) :/
        if (dataSet == EmnistDataSetIterator.Set.LETTERS) {
            oneIndexed = true;
        } else {
            oneIndexed = false;
        }
        this.fOrder = true; //MNIST is C order, EMNIST is F order
        man.close();
    }

    private boolean emnistExists(EmnistFetcher e) {
        //Check 4 files:
        String EMNIST_ROOT = DL4JResources.getDirectory(ResourceType.DATASET, "EMNIST").getAbsolutePath();
        File f = new File(EMNIST_ROOT, e.getTrainingFilesFilename_unzipped());
        if (!f.exists())
            return false;
        f = new File(EMNIST_ROOT, e.getTrainingFileLabelsFilename_unzipped());
        if (!f.exists())
            return false;
        f = new File(EMNIST_ROOT, e.getTestFilesFilename_unzipped());
        if (!f.exists())
            return false;
        f = new File(EMNIST_ROOT, e.getTestFileLabelsFilename_unzipped());
        if (!f.exists())
            return false;
        return true;
    }
}
