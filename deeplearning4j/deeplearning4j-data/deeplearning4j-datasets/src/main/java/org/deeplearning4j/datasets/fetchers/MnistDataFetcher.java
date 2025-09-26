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

import lombok.SneakyThrows;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.base.MnistFetcher;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.deeplearning4j.datasets.mnist.MnistManager;
import org.eclipse.deeplearning4j.resources.DataSetResource;
import org.eclipse.deeplearning4j.resources.ResourceDataSets;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.common.util.MathUtils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.zip.Adler32;
import java.util.zip.Checksum;


public class MnistDataFetcher extends BaseDataFetcher {
    public static final int NUM_EXAMPLES = 60000;
    public static final int NUM_EXAMPLES_TEST = 10000;

    protected static final long CHECKSUM_TRAIN_FEATURES = 2094436111L;
    protected static final long CHECKSUM_TRAIN_LABELS = 4008842612L;
    protected static final long CHECKSUM_TEST_FEATURES = 2165396896L;
    protected static final long CHECKSUM_TEST_LABELS = 2212998611L;

    protected static final long[] CHECKSUMS_TRAIN = new long[]{CHECKSUM_TRAIN_FEATURES, CHECKSUM_TRAIN_LABELS};
    protected static final long[] CHECKSUMS_TEST = new long[]{CHECKSUM_TEST_FEATURES, CHECKSUM_TEST_LABELS};

    protected boolean binarize = true;
    protected boolean train;
    protected int[] order;
    protected Random rng;
    protected boolean shuffle;
    protected boolean oneIndexed = false;
    protected boolean fOrder = false; //MNIST is C order, EMNIST is F order

    protected boolean firstShuffle = true;
    protected  int numExamples = 0;
    protected String images,labels;
    //note: we default to zero here on purpose, otherwise when first initializes an error is thrown.
    private long lastCursor = 0;
    protected MnistManager manager;

    /**
     * Constructor telling whether to binarize the dataset or not
     * @param binarize whether to binarize the dataset or not
     * @throws IOException
     */
    public MnistDataFetcher(boolean binarize) throws IOException {
        this(binarize, true, true, System.currentTimeMillis(), NUM_EXAMPLES);
    }



    public MnistDataFetcher(boolean binarize, boolean train, boolean shuffle, long rngSeed, int numExamples,File topLevelDir) throws IOException {
        if(this instanceof EmnistDataFetcher)
            return;


        this.topLevelDir = topLevelDir;
        long[] checksums;
        DataSetResource imageResource = null;
        DataSetResource labelResource = null;
        if (train) {
            imageResource = topLevelDir() != null ?  ResourceDataSets.mnistTrain(topLevelDir()) :  ResourceDataSets.mnistTrain();
            if(!imageResource.existsLocally())
                imageResource.download(true,3,200000,20000);

            labelResource = topLevelDir() != null ? ResourceDataSets.mnistTrainLabels(topLevelDir()) : ResourceDataSets.mnistTrainLabels();
            if(!labelResource.existsLocally())
                labelResource.download(true,3,200000,20000);

            totalExamples = NUM_EXAMPLES;
            checksums = CHECKSUMS_TRAIN;
        } else {
            imageResource = topLevelDir() != null ?  ResourceDataSets.mnistTest(topLevelDir()) :  ResourceDataSets.mnistTest();
            if(!imageResource.existsLocally())
                imageResource.download(true,3,200000,20000);

            labelResource = topLevelDir() != null ? ResourceDataSets.mnistTestLabels(topLevelDir()) : ResourceDataSets.mnistTestLabels();
            if(!labelResource.existsLocally())
                labelResource.download(true,3,200000,20000);

            totalExamples = NUM_EXAMPLES_TEST;
            checksums = CHECKSUMS_TEST;
        }

        images = imageResource.localPath().getAbsolutePath();
        labels = labelResource.localPath().getAbsolutePath();


        try {
            manager = new MnistManager(images, labels, train);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


        numOutcomes = 10;
        this.binarize = binarize;
        cursor = 0;
        inputColumns = manager.getImages().getEntryLength();
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
        this.numExamples = numExamples;
        reset(); //Shuffle order
    }

    public MnistDataFetcher(boolean binarize, boolean train, boolean shuffle, long rngSeed, int numExamples) throws IOException {
        this(binarize,train,shuffle,rngSeed,numExamples,null);
    }



    private void validateFiles(String[] files, long[] checksums) {
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

    private float[][] featureData = null;

    @SneakyThrows
    @Override
    public void fetch(int numExamples) {
        if (!hasMore()) {
            throw new IllegalStateException("Unable to get more; there are no more images");
        }

        manager.setCurrent((int) lastCursor);
        INDArray labels = Nd4j.zeros(DataType.FLOAT, numExamples, numOutcomes);

        if(featureData == null || featureData.length < numExamples){
            featureData = new float[numExamples][28 * 28];
        }

        int actualExamples = 0;
        byte[] working = null;
        for (int i = 0; i < numExamples; i++, cursor++) {
            if (!hasMore())
                break;

            manager.setCurrent(cursor);
            lastCursor = cursor;
            byte[] img = manager.readImageUnsafe(order[cursor]);

            if (fOrder) {
                //EMNIST requires F order to C order
                if (working == null) {
                    working = new byte[28 * 28];
                }
                for (int j = 0; j < 28 * 28; j++) {
                    working[j] = img[28 * (j % 28) + j / 28];
                }
                img = working;
            }

            int label = manager.readLabel(order[cursor]);
            if (oneIndexed) {
                //For some inexplicable reason, Emnist LETTERS set is indexed 1 to 26 (i.e., 1 to nClasses), while everything else
                // is indexed (0 to nClasses-1) :/
                label--;
            }

            labels.put(actualExamples, label, 1.0f);

            for(int j = 0 ; j < img.length ; j++) {
                featureData[actualExamples][j] = ((int) img[j]) & 0xFF;
            }

            actualExamples++;
        }

        INDArray features;

        if(featureData.length == actualExamples){
            features = Nd4j.create(featureData);
        } else {
            features = Nd4j.create(Arrays.copyOfRange(featureData, 0, actualExamples));
        }

        if (actualExamples < numExamples) {
            labels = labels.get(NDArrayIndex.interval(0, actualExamples), NDArrayIndex.all());
        }

        if(binarize){
            features = features.gt(30.0).castTo(DataType.FLOAT);
        } else {
            features.divi(255.0);
        }

        curr = new DataSet(features, labels);
    }

    @Override
    public void reset() {
        cursor = 0;
        curr = null;
        if (shuffle) {
            if((train && numExamples < NUM_EXAMPLES) || (!train && numExamples < NUM_EXAMPLES_TEST)){
                //Shuffle only first N elements
                if(firstShuffle){
                    MathUtils.shuffleArray(order, rng);
                    firstShuffle = false;
                } else {
                    MathUtils.shuffleArraySubset(order, numExamples, rng);
                }
            } else {
                MathUtils.shuffleArray(order, rng);
            }
        }
    }

    @Override
    public DataSet next() {
        DataSet next = super.next();
        return next;
    }

    public void close() {
    }

}
