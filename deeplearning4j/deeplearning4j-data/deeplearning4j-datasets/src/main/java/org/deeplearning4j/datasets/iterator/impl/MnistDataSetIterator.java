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

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;

import java.io.IOException;

/**
 * MNIST data set iterator - 60000 training digits, 10000 test digits, 10 classes.
 * Digits have 28x28 pixels and 1 channel (grayscale).<br>
 * Produces data in c-order "flattened" format, with shape {@code [minibatch, 784]}<br>
 * For futher details, see <a href="http://yann.lecun.com/exdb/mnist/">http://yann.lecun.com/exdb/mnist/</a>
 *
 * @author Adam Gibson
 * @see EmnistDataSetIterator
 */
public class MnistDataSetIterator extends BaseDatasetIterator {

    public MnistDataSetIterator(int batch, int numExamples) throws IOException {
        this(batch, numExamples, false);
    }

    /**Get the specified number of examples for the MNIST training data set.
     * @param batch the batch size of the examples
     * @param numExamples the overall number of examples
     * @param binarize whether to binarize mnist or not
     * @throws IOException
     */
    public MnistDataSetIterator(int batch, int numExamples, boolean binarize) throws IOException {
        this(batch, numExamples, binarize, true, false, 0);
    }

    /** Constructor to get the full MNIST data set (either test or train sets) without binarization (i.e., just normalization
     * into range of 0 to 1), with shuffling based on a random seed.
     * @param batchSize
     * @param train
     * @throws IOException
     */
    public MnistDataSetIterator(int batchSize, boolean train, int seed) throws IOException {
        this(batchSize, (train ? MnistDataFetcher.NUM_EXAMPLES : MnistDataFetcher.NUM_EXAMPLES_TEST), false, train,
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
    public MnistDataSetIterator(int batch, int numExamples, boolean binarize, boolean train, boolean shuffle,
                    long rngSeed) throws IOException {
        super(batch, numExamples, new MnistDataFetcher(binarize, train, shuffle, rngSeed, numExamples));
    }
}
