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

import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.fetchers.UciSequenceDataFetcher;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

/**
 * UCI synthetic control chart time series dataset. This dataset is useful for classification of univariate
 * time series with six categories:<br>
 * Normal, Cyclic, Increasing trend, Decreasing trend, Upward shift, Downward shift
 *
 * Details:     <a href="https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series">https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series</a><br>
 * Data:        <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data">https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data</a><br>
 * Image:       <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/data.jpeg">https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/data.jpeg</a>
 *
 * @author Briton Park (bpark738)
 */
public class UciSequenceDataSetIterator extends SequenceRecordReaderDataSetIterator {

    protected DataSetPreProcessor preProcessor;

    /**
     * Create an iterator for the training set, with the specified minibatch size. Randomized with RNG seed 123
     *
     * @param batchSize Minibatch size
     */
    public UciSequenceDataSetIterator(int batchSize) {
        this(batchSize, DataSetType.TRAIN, 123);
    }

    /**
     * Create an iterator for the training or test set, with the specified minibatch size. Randomized with RNG seed 123
     *
     * @param batchSize Minibatch size
     * @param set       Set: training or test
     */
    public UciSequenceDataSetIterator(int batchSize, DataSetType set) {
        this(batchSize, set, 123);
    }

    /**
     * Create an iterator for the training or test set, with the specified minibatch size
     *
     * @param batchSize Minibatch size
     * @param set       Set: training or test
     * @param rngSeed   Random number generator seed to use for randomization
     */
    public UciSequenceDataSetIterator(int batchSize, DataSetType set, long rngSeed) {
        super(new UciSequenceDataFetcher().getRecordReader(rngSeed, set), batchSize, UciSequenceDataFetcher.NUM_LABELS, 1);
        // last parameter is index of label
    }
}