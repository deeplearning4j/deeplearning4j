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

package org.nd4j.linalg.dataset.api.iterator;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

import java.util.ArrayList;
import java.util.List;

/**
 * Splits a dataset (represented as a single DataSet object) into k folds.
 * DataSet is duplicated in memory once
 * call .next() to get the k-1 folds to train on and call .testfold() to get the corresponding kth fold for testing
 * @author Susan Eraly
 */
public class KFoldIterator implements DataSetIterator {
    private DataSet allData;
    private int k;
    private int batch;
    private int lastBatch;
    private int kCursor = 0;
    private DataSet test;
    private DataSet train;
    protected DataSetPreProcessor preProcessor;

    public KFoldIterator(DataSet allData) {
        this(10, allData);
    }

    /**Create an iterator given the dataset and a value of k (optional, defaults to 10)
     * If number of samples in the dataset is not a multiple of k, the last fold will have less samples with the rest having the same number of samples.
     *
     * @param k number of folds (optional, defaults to 10)
     * @param allData DataSet to split into k folds
     */

    public KFoldIterator(int k, DataSet allData) {
        this.k = k;
        this.allData = allData.copy();
        if (k <= 1)
            throw new IllegalArgumentException();
        if (allData.numExamples() % k != 0) {
            if (k != 2) {
                this.batch = allData.numExamples() / (k - 1);
                this.lastBatch = allData.numExamples() % (k - 1);
            } else {
                this.lastBatch = allData.numExamples() / 2;
                this.batch = this.lastBatch + 1;
            }
        } else {
            this.batch = allData.numExamples() / k;
            this.lastBatch = allData.numExamples() / k;
        }
    }

    @Override
    public DataSet next(int num) throws UnsupportedOperationException {
        return null;
    }

    /**
     * Returns total number of examples in the dataset (all k folds)
     *
     * @return total number of examples in the dataset including all k folds
     */
    public int totalExamples() {
        // FIXME: int cast
        return (int) allData.getLabels().size(0);
    }

    @Override
    public int inputColumns() {
        // FIXME: int cast
        return (int) allData.getFeatures().size(1);
    }

    @Override
    public int totalOutcomes() {
        // FIXME: int cast
        return (int) allData.getLabels().size(1);
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    /**
     * Shuffles the dataset and resets to the first fold
     *
     * @return void
     */
    @Override
    public void reset() {
        //shuffle and return new k folds
        allData.shuffle();
        kCursor = 0;
    }


    /**
     * The number of examples in every fold, except the last if totalexamples % k !=0
     *
     * @return examples in a fold
     */
    @Override
    public int batch() {
        return batch;
    }

    /**
     * The number of examples in the last fold
     * if totalexamples % k == 0 same as the number of examples in every other fold
     *
     * @return examples in the last fold
     */
    public int lastBatch() {
        return lastBatch;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return allData.getLabelNamesList();
    }

    @Override
    public boolean hasNext() {
        return kCursor < k;
    }

    @Override
    public DataSet next() {
        nextFold();
        return train;
    }

    @Override
    public void remove() {
        // no-op
    }

    private void nextFold() {
        int left;
        int right;
        if (kCursor == k - 1) {
            left = totalExamples() - lastBatch;
            right = totalExamples();
        } else {
            left = kCursor * batch;
            right = left + batch;
        }

        List<DataSet> kMinusOneFoldList = new ArrayList<DataSet>();
        if (right < totalExamples()) {
            if (left > 0) {
                kMinusOneFoldList.add((DataSet) allData.getRange(0, left));
            }
            kMinusOneFoldList.add((DataSet) allData.getRange(right, totalExamples()));
            train = DataSet.merge(kMinusOneFoldList);
        } else {
            train = (DataSet) allData.getRange(0, left);
        }
        test = (DataSet) allData.getRange(left, right);

        kCursor++;

    }

    /**
     * @return the held out fold as a dataset
     */
    public DataSet testFold() {
        return test;
    }
}
