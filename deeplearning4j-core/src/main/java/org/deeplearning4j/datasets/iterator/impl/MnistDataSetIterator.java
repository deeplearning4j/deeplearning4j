/*
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

import java.io.IOException;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

/**
 * Mnist data applyTransformToDestination iterator.
 * @author Adam Gibson
 */
public class MnistDataSetIterator extends BaseDatasetIterator {

	public MnistDataSetIterator(int batch,int numExamples) throws IOException {
		this(batch,numExamples,false);
	}

    /**
     * Whether to binarize the data or not
     * @param batch the the batch size of the examples
     * @param numExamples the overall number of examples
     * @param binarize whether to binarize mnist or not
     * @throws IOException
     */
    public MnistDataSetIterator(int batch, int numExamples, boolean binarize) throws IOException {
        this(batch,numExamples,binarize,true,false,System.currentTimeMillis());
    }

    public MnistDataSetIterator(int batchSize, boolean train) throws IOException{
        this(batchSize, (train ? MnistDataFetcher.NUM_EXAMPLES : MnistDataFetcher.NUM_EXAMPLES_TEST), false, train, true, System.currentTimeMillis());
    }

    /**
     *
     * @param batch Size of each patch
     * @param numExamples total number of examples to load
     * @param binarize whether to binarize the data or not (if false: normalize in range 0 to 1)
     * @param train Train vs. test set
     * @param shuffle whether to shuffle the examples
     * @param rngSeed random number generator seed to use when shuffling examples
     */
    public MnistDataSetIterator(int batch, int numExamples, boolean binarize, boolean train, boolean shuffle, long rngSeed) throws IOException {
        super(batch, numExamples,new MnistDataFetcher(binarize,train,shuffle,rngSeed));
    }


}
