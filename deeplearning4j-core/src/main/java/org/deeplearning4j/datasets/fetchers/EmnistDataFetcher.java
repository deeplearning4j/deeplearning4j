/*-
 *
 *  * Copyright 2015 Skymind,Inc.
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

package org.deeplearning4j.datasets.fetchers;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.base.EmnistFetcher;
import org.deeplearning4j.base.MnistFetcher;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.MnistManager;
import org.deeplearning4j.util.MathUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;


/**
 * Data fetcher for the EMNIST dataset
 *
 * @author Alex Black
 *
 */
public class EmnistDataFetcher extends MnistDataFetcher {
    protected static final String TEMP_ROOT = System.getProperty("user.home");
    protected static final String EMNIST_ROOT = TEMP_ROOT + File.separator + "EMNIST" + File.separator;

    protected transient MnistManager man;
    protected boolean binarize = true;
    protected boolean train;
    protected int[] order;
    protected Random rng;
    protected boolean shuffle;

    public EmnistDataFetcher(EmnistDataSetIterator.DataSet dataSet, boolean binarize, boolean train, boolean shuffle, long rngSeed) throws IOException {
        if (!emnistExists(dataSet)) {
            new EmnistFetcher(dataSet).downloadAndUntar();
        }
        String images;
        String labels;
        if (train) {
            images = EMNIST_ROOT + MnistFetcher.trainingFilesFilename_unzipped;
            labels = EMNIST_ROOT + MnistFetcher.trainingFileLabelsFilename_unzipped;
            totalExamples = EmnistDataSetIterator.numExamplesTrain(dataSet);
        } else {
            images = EMNIST_ROOT + MnistFetcher.testFilesFilename_unzipped;
            labels = EMNIST_ROOT + MnistFetcher.testFileLabelsFilename_unzipped;
            totalExamples = EmnistDataSetIterator.numExamplesTest(dataSet);
        }

        try {
            man = new MnistManager(images, labels, train);
        } catch (Exception e) {
            FileUtils.deleteDirectory(new File(EMNIST_ROOT));
            new MnistFetcher().downloadAndUntar();
            man = new MnistManager(images, labels, train);
        }

        numOutcomes = 10;
        this.binarize = binarize;
        cursor = 0;
        inputColumns = man.getImages().getEntryLength();
        this.train = train;
        this.shuffle = shuffle;

        order = new int[totalExamples];
        for (int i = 0; i < order.length; i++)
            order[i] = i;
        rng = new Random(rngSeed);
        reset(); //Shuffle order
    }

    private boolean emnistExists(EmnistDataSetIterator.DataSet dataSet) {
        EmnistFetcher e = new EmnistFetcher(dataSet);

        //Check 4 files:
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

    @Override
    public void fetch(int numExamples) {
        if (!hasMore()) {
            throw new IllegalStateException("Unable to getFromOrigin more; there are no more images");
        }


        float[][] featureData = new float[numExamples][0];
        float[][] labelData = new float[numExamples][0];

        int actualExamples = 0;
        for (int i = 0; i < numExamples; i++, cursor++) {
            if (!hasMore())
                break;

            byte[] img = man.readImageUnsafe(order[cursor]);
            int label = man.readLabel(order[cursor]);

            float[] featureVec = new float[img.length];
            featureData[actualExamples] = featureVec;
            labelData[actualExamples] = new float[10];
            labelData[actualExamples][label] = 1.0f;

            for (int j = 0; j < img.length; j++) {
                float v = ((int) img[j]) & 0xFF; //byte is loaded as signed -> convert to unsigned
                if (binarize) {
                    if (v > 30.0f)
                        featureVec[j] = 1.0f;
                    else
                        featureVec[j] = 0.0f;
                } else {
                    featureVec[j] = v / 255.0f;
                }
            }

            actualExamples++;
        }

        if (actualExamples < numExamples) {
            featureData = Arrays.copyOfRange(featureData, 0, actualExamples);
            labelData = Arrays.copyOfRange(labelData, 0, actualExamples);
        }

        INDArray features = Nd4j.create(featureData);
        INDArray labels = Nd4j.create(labelData);
        curr = new DataSet(features, labels);
    }

    @Override
    public void reset() {
        cursor = 0;
        curr = null;
        if (shuffle)
            MathUtils.shuffleArray(order, rng);
    }

    @Override
    public DataSet next() {
        DataSet next = super.next();
        return next;
    }

}
