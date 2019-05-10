/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.datasets.iterator.impl;

import lombok.Getter;
import org.deeplearning4j.datasets.fetchers.EmnistDataFetcher;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * EMNIST DataSetIterator<br>
 * EMNIST is similar to the common MNIST dataset (available via {@link MnistDataSetIterator}), with 6 different splits/
 * variants, specified by {@link Set}:<br>
 * <ul>
 *     <li>COMPLETE: Also known as 'ByClass' split. 814,255 examples total (train + test), 62 classes</li>
 *     <li>MERGE: Also known as 'ByMerge' split. 814,255 examples total. 47 unbalanced classes. Combines lower and upper
 *     case characters (that are difficult to distinguish) into one class for each letter (instead of 2), for letters
 *     C, I, J, K, L, M, O, P, S, U, V, W, X, Y and Z </li>
 *     <li>BALANCED: 131,600 examples total. 47 classes (equal number of examples in each class)</li>
 *     <li>LETTERS: 145,600 examples total. 26 balanced classes</li>
 *     <li>DIGITS: 280,000 examples total. 10 balanced classes</li>
 *     <li>MNIST: 70,000 examples total. 10 balanced classes. Equivalent to the original MNIST dataset in {@link MnistDataSetIterator}</li>
 * </ul>
 * <br>
 * See: <a href="https://www.nist.gov/itl/iad/image-group/emnist-dataset">
 *     https://www.nist.gov/itl/iad/image-group/emnist-dataset</a> and
 * <a href="https://arxiv.org/abs/1702.05373">https://arxiv.org/abs/1702.05373</a>
 *
 * As per {@link MnistDataSetIterator}, the features data is in "flattened" format: shape [minibatch, 784].
 *
 * @author Alex Black
 */
public class EmnistDataSetIterator extends BaseDatasetIterator {

    private static final int NUM_COMPLETE_TRAIN = 697932;
    private static final int NUM_COMPLETE_TEST = 116323;

    private static final int NUM_MERGE_TRAIN = 697932;
    private static final int NUM_MERGE_TEST = 116323;

    private static final int NUM_BALANCED_TRAIN = 112800;
    private static final int NUM_BALANCED_TEST = 18800;

    private static final int NUM_DIGITS_TRAIN = 240000;
    private static final int NUM_DIGITS_TEST = 40000;

    private static final int NUM_LETTERS_TRAIN = 88800;
    private static final int NUM_LETTERS_TEST = 14800;

    private static final int NUM_MNIST_TRAIN = 60000;
    private static final int NUM_MNIST_TEST = 10000;

    private static final char[] LABELS_COMPLETE = new char[] {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68,
                    69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99,
                    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                    120, 121, 122};

    private static final char[] LABELS_MERGE = new char[] {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69,
                    70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 100,
                    101, 102, 103, 104, 110, 113, 114, 116};

    private static final char[] LABELS_BALANCED = new char[] {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68,
                    69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 100,
                    101, 102, 103, 104, 110, 113, 114, 116};

    private static final char[] LABELS_DIGITS = new char[] {48, 49, 50, 51, 52, 53, 54, 55, 56, 57};

    private static final char[] LABELS_LETTERS = new char[] {65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90};

    /**
     * EMNIST dataset has multiple different subsets. See {@link EmnistDataSetIterator} Javadoc for details.
     */
    public enum Set {
        COMPLETE, MERGE, BALANCED, LETTERS, DIGITS, MNIST
    }

    protected Set dataSet;
    protected int batch, numExamples;
    @Getter
    protected DataSetPreProcessor preProcessor;

    /**
     * Create an EMNIST iterator with randomly shuffled data based on a random RNG seed
     *
     * @param dataSet Dataset (subset) to return
     * @param batch   Batch size
     * @param train   If true: use training set. If false: use test set
     * @throws IOException If an error occurs when loading/downloading the dataset
     */
    public EmnistDataSetIterator(Set dataSet, int batch, boolean train) throws IOException {
        this(dataSet, batch, train, System.currentTimeMillis());
    }

    /**
     * Create an EMNIST iterator with randomly shuffled data based on a specified RNG seed
     *
     * @param dataSet   Dataset (subset) to return
     * @param batchSize Batch size
     * @param train     If true: use training set. If false: use test set
     * @param seed      Random number generator seed
     */
    public EmnistDataSetIterator(Set dataSet, int batchSize, boolean train, long seed) throws IOException {
        this(dataSet, batchSize, false, train, true, seed);
    }

    /**
     * Get the specified number of MNIST examples (test or train set), with optional shuffling and binarization.
     *
     * @param batch    Size of each minibatch
     * @param binarize whether to binarize the data or not (if false: normalize in range 0 to 1)
     * @param train    Train vs. test set
     * @param shuffle  whether to shuffle the examples
     * @param rngSeed  random number generator seed to use when shuffling examples
     */
    public EmnistDataSetIterator(Set dataSet, int batch, boolean binarize, boolean train, boolean shuffle, long rngSeed)
            throws IOException {
        super(batch, numExamples(train, dataSet), new EmnistDataFetcher(dataSet, binarize, train, shuffle, rngSeed));
        this.dataSet = dataSet;
    }

    private static int numExamples(boolean train, Set ds) {
        if (train) {
            return numExamplesTrain(ds);
        } else {
            return numExamplesTest(ds);
        }
    }

    /**
     * Get the number of training examples for the specified subset
     *
     * @param dataSet Subset to get
     * @return Number of examples for the specified subset
     */
    public static int numExamplesTrain(Set dataSet) {
        switch (dataSet) {
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

    /**
     * Get the number of test examples for the specified subset
     *
     * @param dataSet Subset to get
     * @return Number of examples for the specified subset
     */
    public static int numExamplesTest(Set dataSet) {
        switch (dataSet) {
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

    /**
     * Get the number of labels for the specified subset
     *
     * @param dataSet Subset to get
     * @return Number of labels for the specified subset
     */
    public static int numLabels(Set dataSet) {
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

    /**
     * Get the labels as a character array
     *
     * @return Labels
     */
    public char[] getLabelsArrays() {
        return getLabelsArray(dataSet);
    }

    /**
     * Get the labels as a List<String>
     *
     * @return Labels
     */
    public List<String> getLabels() {
        return getLabels(dataSet);
    }

    /**
     * Get the label assignments for the given set as a character array.
     *
     * @param dataSet DataSet to get the label assignment for
     * @return Label assignment and given dataset
     */
    public static char[] getLabelsArray(Set dataSet) {
        switch (dataSet) {
            case COMPLETE:
                return LABELS_COMPLETE;
            case MERGE:
                return LABELS_MERGE;
            case BALANCED:
                return LABELS_BALANCED;
            case LETTERS:
                return LABELS_LETTERS;
            case DIGITS:
            case MNIST:
                return LABELS_DIGITS;
            default:
                throw new UnsupportedOperationException("Unknown Set: " + dataSet);
        }
    }

    /**
     * Get the label assignments for the given set as a List<String>
     *
     * @param dataSet DataSet to get the label assignment for
     * @return Label assignment and given dataset
     */
    public static List<String> getLabels(Set dataSet) {
        char[] c = getLabelsArray(dataSet);
        List<String> l = new ArrayList<>(c.length);
        for (char c2 : c) {
            l.add(String.valueOf(c2));
        }
        return l;
    }

    /**
     * Are the labels balanced in the training set (that is: are the number of examples for each label equal?)
     *
     * @param dataSet Set to get balanced value for
     * @return True if balanced dataset, false otherwise
     */
    public static boolean isBalanced(Set dataSet) {
        switch (dataSet) {
            case COMPLETE:
            case MERGE:
            case LETTERS:
                //Note: EMNIST docs claims letters is balanced, but this is not possible for training set:
                // 88800 examples / 26 classes = 3418.46
                return false;
            case BALANCED:
            case DIGITS:
            case MNIST:
                return true;
            default:
                throw new UnsupportedOperationException("Unknown Set: " + dataSet);
        }
    }
}
