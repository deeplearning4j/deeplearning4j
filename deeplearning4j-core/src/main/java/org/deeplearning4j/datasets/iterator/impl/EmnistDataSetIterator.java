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

package org.deeplearning4j.datasets.iterator.impl;

import lombok.Getter;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.deeplearning4j.datasets.fetchers.EmnistDataFetcher;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

import java.io.IOException;

/**
 * EMNIST DataSetIterator<br>
 * EMNIST is similar to the common MNIST dataset (available via {@link MnistDataSetIterator}), with 6 different splits/
 * variants:<br>
 * <ul></ul>
 * https://www.nist.gov/itl/iad/image-group/emnist-dataset<br>
 * https://arxiv.org/abs/1702.05373<br>
 *
 * @author Alex Black
 */
public class EmnistDataSetIterator extends BaseDatasetIterator {

    public static final int NUM_COMPLETE_TRAIN = 697932;
    public static final int NUM_COMPLETE_TEST = 116323;

    public static final int NUM_MERGE_TRAIN = 697932;
    public static final int NUM_MERGE_TEST = 116323;

    public static final int NUM_BALANCED_TRAIN = 112800;
    public static final int NUM_BALANCED_TEST = 18800;

    public static final int NUM_DIGITS_TRAIN = 240000;
    public static final int NUM_DIGITS_TEST = 40000;

    public static final int NUM_LETTERS_TRAIN = 88800;
    public static final int NUM_LETTERS_TEST = 14800;

    public static final int NUM_MNIST_TRAIN = 60000;
    public static final int NUM_MNIST_TEST = 10000;


    public enum Set {
        COMPLETE,
        MERGE,
        BALANCED,
        LETTERS,
        DIGITS,
        MNIST
    }

    protected int batch, numExamples;
    protected BaseDataFetcher fetcher;
    @Getter
    protected DataSetPreProcessor preProcessor;

    public EmnistDataSetIterator(Set dataSet, int batch, boolean train) throws IOException {
        this(dataSet, batch, train, System.currentTimeMillis());
    }

    public EmnistDataSetIterator(Set dataSet, int batchSize, boolean train, long seed) throws IOException {
        this(dataSet, batchSize,false, train, false, seed);
    }

    /**Get the specified number of MNIST examples (test or train set), with optional shuffling and binarization.
     * @param batch Size of each patch
     * @param binarize whether to binarize the data or not (if false: normalize in range 0 to 1)
     * @param train Train vs. test set
     * @param shuffle whether to shuffle the examples
     * @param rngSeed random number generator seed to use when shuffling examples
     */
    public EmnistDataSetIterator(Set dataSet, int batch, boolean binarize, boolean train, boolean shuffle,
                                 long rngSeed) throws IOException {
        super(batch, numExamples(train, dataSet), new EmnistDataFetcher(dataSet, binarize, train, shuffle, rngSeed));
    }

    private static int numExamples(boolean train, Set ds){
        if(train){
            return numExamplesTrain(ds);
        } else {
            return numExamplesTest(ds);
        }
    }


    public static int numExamplesTrain(Set dataSet){
        switch (dataSet){
            case COMPLETE:
                return NUM_COMPLETE_TRAIN;
            case MERGE:
                return NUM_MERGE_TRAIN;
            case BALANCED:
                return NUM_BALANCED_TRAIN;
            case LETTERS:
                return NUM_LETTERS_TRAIN;
            case DIGITS:
                return NUM_DIGITS_TRAIN;
            case MNIST:
                return NUM_MNIST_TRAIN;
            default:
                throw new UnsupportedOperationException("Unknown Set: " + dataSet);
        }
    }

    public static int numExamplesTest(Set dataSet){
        switch (dataSet){
            case COMPLETE:
                return NUM_COMPLETE_TEST;
            case MERGE:
                return NUM_MERGE_TEST;
            case BALANCED:
                return NUM_BALANCED_TEST;
            case LETTERS:
                return NUM_LETTERS_TEST;
            case DIGITS:
                return NUM_DIGITS_TEST;
            case MNIST:
                return NUM_MNIST_TEST;
            default:
                throw new UnsupportedOperationException("Unknown Set: " + dataSet);
        }
    }

    public static int numLabels(Set dataSet){
        switch (dataSet){
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

    public static int numExamplesTotal(Set dataSet){
        return numExamplesTrain(dataSet) + numExamplesTest(dataSet);
    }
}
