/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.dataset.api.iterator;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

import java.util.ArrayList;
import java.util.List;

/**
 * Splits a dataset (represented as a single DataSet object) into k folds.
 * DataSet is duplicated in memory once.
 * Call .next() to get the k-1 folds to train on and then call .testfold() to get the corresponding kth fold for testing
 * @author Susan Eraly
 * @author Tamas Fenyvesi - modified KFoldIterator following the scikit-learn implementation (December 2018)
 */
public class KFoldIterator implements DataSetIterator {
	
	private static final long serialVersionUID = 6130298603412865817L;
	
	protected DataSet allData;
    protected int k;
    protected int N;
    protected int[] intervalBoundaries;
    protected int kCursor = 0;
    protected DataSet test;
    protected DataSet train;
    protected DataSetPreProcessor preProcessor;

    /**
     * Create a k-fold cross-validation iterator given the dataset and k=10 train-test splits.
     * N number of samples are split into k batches. The first (N%k) batches contain (N/k)+1 samples, while the remaining batches contain (N/k) samples. 
     * In case the number of samples (N) in the dataset is a multiple of k, all batches will contain (N/k) samples.
     * @param allData DataSet to split into k folds
     */
    public KFoldIterator(DataSet allData) {
        this(10, allData);
    }

    /**
     * Create an iterator given the dataset with given k train-test splits
     * N number of samples are split into k batches. The first (N%k) batches contain (N/k)+1 samples, while the remaining batches contain (N/k) samples.
     * In case the number of samples (N) in the dataset is a multiple of k, all batches will contain (N/k) samples.
     * @param k number of folds (optional, defaults to 10)
     * @param allData DataSet to split into k folds
     */
    public KFoldIterator(int k, DataSet allData) {
        if (k <= 1) {
            throw new IllegalArgumentException();
        }
        this.k = k;
        this.N = allData.numExamples();
        this.allData = allData;
        
        // generate index interval boundaries of test folds
        int baseBatchSize = N / k;
        int numIncrementedBatches = N % k;

        this.intervalBoundaries = new int[k+1];
        intervalBoundaries[0] = 0;
        for (int i = 1; i <= k; i++) {
        	if (i <= numIncrementedBatches) {
                intervalBoundaries[i] = intervalBoundaries[i-1] + (baseBatchSize + 1);
            } else {
            	intervalBoundaries[i] = intervalBoundaries[i-1] + baseBatchSize;
            }
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
        return N;
    }

    @Override
    public int inputColumns() {
        return (int) allData.getFeatures().size(1);
    }

    @Override
    public int totalOutcomes() {
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
     * The number of examples in every fold is (N / k), 
     * except when (N % k) > 0, when the first (N % k) folds contain (N / k) + 1 examples  
     *
     * @return examples in a fold
     */
    @Override
    public int batch() {
    	return intervalBoundaries[kCursor+1] - intervalBoundaries[kCursor];
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

    protected void nextFold() {
        int left = intervalBoundaries[kCursor];
        int right = intervalBoundaries[kCursor + 1];

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
