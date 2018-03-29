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
import org.deeplearning4j.base.MnistFetcher;
import org.deeplearning4j.datasets.mnist.MnistManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.MathUtils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.zip.Adler32;
import java.util.zip.Checksum;


/**
 * Data fetcher for the MNIST dataset
 * @author Adam Gibson
 *
 */
public class MnistDataFetcher extends BaseDataFetcher {
    public static final int NUM_EXAMPLES = 60000;
    public static final int NUM_EXAMPLES_TEST = 10000;
    protected static final String TEMP_ROOT = System.getProperty("user.home");
    protected static final String MNIST_ROOT = TEMP_ROOT + File.separator + "MNIST" + File.separator;

    protected static final long CHECKSUM_TRAIN_FEATURES = 2094436111L;
    protected static final long CHECKSUM_TRAIN_LABELS = 4008842612L;
    protected static final long CHECKSUM_TEST_FEATURES = 2165396896L;
    protected static final long CHECKSUM_TEST_LABELS = 2212998611L;

    protected static final long[] CHECKSUMS_TRAIN = new long[]{CHECKSUM_TRAIN_FEATURES, CHECKSUM_TRAIN_LABELS};
    protected static final long[] CHECKSUMS_TEST = new long[]{CHECKSUM_TEST_FEATURES, CHECKSUM_TEST_LABELS};

    protected transient MnistManager man;
    protected boolean binarize = true;
    protected boolean train;
    protected int[] order;
    protected Random rng;
    protected boolean shuffle;
    protected boolean oneIndexed = false;
    protected boolean fOrder = false; //MNIST is C order, EMNIST is F order


    /**
     * Constructor telling whether to binarize the dataset or not
     * @param binarize whether to binarize the dataset or not
     * @throws IOException
     */
    public MnistDataFetcher(boolean binarize) throws IOException {
        this(binarize, true, true, System.currentTimeMillis());
    }

    public MnistDataFetcher(boolean binarize, boolean train, boolean shuffle, long rngSeed) throws IOException {
        if (!mnistExists()) {
            new MnistFetcher().downloadAndUntar();
        }
        String images;
        String labels;
        long[] checksums;
        if (train) {
            images = MNIST_ROOT + MnistFetcher.TRAINING_FILES_FILENAME_UNZIPPED;
            labels = MNIST_ROOT + MnistFetcher.TRAINING_FILE_LABELS_FILENAME_UNZIPPED;
            totalExamples = NUM_EXAMPLES;
            checksums = CHECKSUMS_TRAIN;
        } else {
            images = MNIST_ROOT + MnistFetcher.TEST_FILES_FILENAME_UNZIPPED;
            labels = MNIST_ROOT + MnistFetcher.TEST_FILE_LABELS_FILENAME_UNZIPPED;
            totalExamples = NUM_EXAMPLES_TEST;
            checksums = CHECKSUMS_TEST;
        }
        String[] files = new String[]{images, labels};

        try {
            man = new MnistManager(images, labels, train);
            validateFiles(files, checksums);
        } catch (Exception e) {
            try {
                FileUtils.deleteDirectory(new File(MNIST_ROOT));
            } catch (Exception e2){ }
            new MnistFetcher().downloadAndUntar();
            man = new MnistManager(images, labels, train);
            validateFiles(files, checksums);
        }

        numOutcomes = 10;
        this.binarize = binarize;
        cursor = 0;
        inputColumns = man.getImages().getEntryLength();
        this.train = train;
        this.shuffle = shuffle;

        if (train) {
            order = new int[NUM_EXAMPLES];
        } else {
            order = new int[NUM_EXAMPLES_TEST];
        }
        for (int i = 0; i < order.length; i++)
            order[i] = i;
        rng = new Random(rngSeed);
        reset(); //Shuffle order
    }

    private boolean mnistExists() {
        //Check 4 files:
        File f = new File(MNIST_ROOT, MnistFetcher.TRAINING_FILES_FILENAME_UNZIPPED);
        if (!f.exists())
            return false;
        f = new File(MNIST_ROOT, MnistFetcher.TRAINING_FILE_LABELS_FILENAME_UNZIPPED);
        if (!f.exists())
            return false;
        f = new File(MNIST_ROOT, MnistFetcher.TEST_FILES_FILENAME_UNZIPPED);
        if (!f.exists())
            return false;
        f = new File(MNIST_ROOT, MnistFetcher.TEST_FILE_LABELS_FILENAME_UNZIPPED);
        if (!f.exists())
            return false;
        return true;
    }

    private void validateFiles(String[] files, long[] checksums){
        //Validate files:
        try {
            for (int i = 0; i < files.length; i++) {
                File f = new File(files[i]);
                Checksum adler = new Adler32();
                long checksum = f.exists() ? FileUtils.checksum(f, adler).getValue() : -1;
                if (!f.exists() || checksum != checksums[i]) {
                    throw new IllegalStateException("Failed checksum: expected " + checksums[i] +
                            ", got " + checksum + " for file: " + f);
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public MnistDataFetcher() throws IOException {
        this(true);
    }

    @Override
    public void fetch(int numExamples) {
        if (!hasMore()) {
            throw new IllegalStateException("Unable to get more; there are no more images");
        }

        float[][] featureData = new float[numExamples][0];
        float[][] labelData = new float[numExamples][0];

        int actualExamples = 0;
        byte[] working = null;
        for (int i = 0; i < numExamples; i++, cursor++) {
            if (!hasMore())
                break;

            byte[] img = man.readImageUnsafe(order[cursor]);

            if (fOrder) {
                //EMNIST requires F order to C order
                if (working == null) {
                    working = new byte[28 * 28];
                }
                for (int j = 0; j < 28 * 28; j++) {
                    working[j] = img[28 * (j % 28) + j / 28];
                }
                byte[] temp = img;
                img = working;
                working = temp;
            }

            int label = man.readLabel(order[cursor]);
            if (oneIndexed) {
                //For some inexplicable reason, Emnist LETTERS set is indexed 1 to 26 (i.e., 1 to nClasses), while everything else
                // is indexed (0 to nClasses-1) :/
                label--;
            }

            float[] featureVec = new float[img.length];
            featureData[actualExamples] = featureVec;
            labelData[actualExamples] = new float[numOutcomes];
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
