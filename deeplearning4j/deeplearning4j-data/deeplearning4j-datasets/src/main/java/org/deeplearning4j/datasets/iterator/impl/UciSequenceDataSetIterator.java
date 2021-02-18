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

package org.deeplearning4j.datasets.iterator.impl;

import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.fetchers.UciSequenceDataFetcher;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

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