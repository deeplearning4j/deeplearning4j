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
 * EMNIST DataSetIterator
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


    public enum DataSet {
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

    public EmnistDataSetIterator(int batch, int numExamples) throws IOException {
        this(DataSet.COMPLETE, batch, numExamples, false);
    }

    /**Get the specified number of examples for the MNIST training data set.
     * @param batch the batch size of the examples
     * @param numExamples the overall number of examples
     * @param binarize whether to binarize mnist or not
     * @throws IOException
     */
    public EmnistDataSetIterator(DataSet dataSet, int batch, int numExamples, boolean binarize) throws IOException {
        this(dataSet, batch, numExamples, binarize, true, false, 0);
    }

    /** Constructor to get the full MNIST data set (either test or train sets) without binarization (i.e., just normalization
     * into range of 0 to 1), with shuffling based on a random seed.
     * @param batchSize
     * @param train
     * @throws IOException
     */
    public EmnistDataSetIterator(DataSet dataSet, int batchSize, boolean train, int seed) throws IOException {
        this(dataSet, batchSize, (train ? MnistDataFetcher.NUM_EXAMPLES : MnistDataFetcher.NUM_EXAMPLES_TEST), false, train,
                        true, seed);
    }

    /**Get the specified number of MNIST examples (test or train set), with optional shuffling and binarization.
     * @param batch Size of each patch
     * @param numExamples total number of examples to load
     * @param binarize whether to binarize the data or not (if false: normalize in range 0 to 1)
     * @param train Train vs. test set
     * @param shuffle whether to shuffle the examples
     * @param rngSeed random number generator seed to use when shuffling examples
     */
    public EmnistDataSetIterator(DataSet dataSet, int batch, int numExamples, boolean binarize, boolean train, boolean shuffle,
                                 long rngSeed) throws IOException {
        super(batch, numExamples, new EmnistDataFetcher(dataSet, binarize, train, shuffle, rngSeed));
    }


    public static int numExamplesTrain(DataSet dataSet){
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
                throw new UnsupportedOperationException("Unknown DataSet: " + dataSet);
        }
    }

    public static int numExamplesTest(DataSet dataSet){
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
                throw new UnsupportedOperationException("Unknown DataSet: " + dataSet);
        }
    }

    public static int numExamplesTotal(DataSet dataSet){
        return numExamplesTrain(dataSet) + numExamplesTest(dataSet);
    }
}
