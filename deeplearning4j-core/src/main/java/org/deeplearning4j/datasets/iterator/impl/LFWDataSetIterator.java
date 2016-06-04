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

import org.canova.api.io.labels.PathLabelGenerator;
import org.canova.image.loader.LFWLoader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;

import java.util.Random;


public class LFWDataSetIterator extends RecordReaderDataSetIterator {

	protected static int height = 250;
	protected static int width = 250;
	protected static int channels = 3;

    /**
     * Create LFW data specific iterator
     * @param imgDim shape of input
     * */
    public LFWDataSetIterator(int[] imgDim) {
        super(new LFWLoader().getRecordReader(imgDim[0], imgDim[1], imgDim[2]), 0, 1, 0);
    }


    /**
	 * Create LFW data specific iterator
	 * @param batchSize the batch size of the examples
	 * @param numExamples the overall number of examples
	 * */
	public LFWDataSetIterator(int batchSize, int numExamples) {
		super(new LFWLoader().getRecordReader(numExamples, batchSize, true), batchSize, 1, LFWLoader.NUM_LABELS);
	}

	/**
	 * Create LFW data specific iterator
	 * @param batchSize the batch size of the examples
	 * @param imgDim an array of height, width and channels
	 * @param numExamples the overall number of examples
	 * */
	public LFWDataSetIterator(int batchSize, int numExamples, int[] imgDim, boolean train) {
		super(new LFWLoader().getRecordReader(numExamples, batchSize, imgDim[0], imgDim[1], imgDim[2], train, new Random(123)), batchSize, 1, LFWLoader.NUM_LABELS);
	}

	/**
	 * Create LFW data specific iterator
	 * @param batchSize the batch size of the examples
	 * @param numExamples the overall number of examples
	 * @param numCategories the overall number of labels
     * @param train true if use train value
	 * */
	public LFWDataSetIterator(int batchSize, int numExamples, int numCategories, boolean train) {
		super(new LFWLoader().getRecordReader(numExamples, batchSize, train), batchSize, 1, numCategories);
	}

	/**
	 * Create LFW data specific iterator
	 * @param batchSize the batch size of the examples
	 * @param imgDim an array of height, width and channels
	 */
	public LFWDataSetIterator(int batchSize, int[] imgDim)  {
		super(new LFWLoader().getRecordReader(imgDim[0], imgDim[1], imgDim[2]), batchSize, 1, LFWLoader.NUM_LABELS);
	}

	/**
	 * Create LFW data specific iterator
	 * @param batchSize the batch size of the examples
	 * @param imgDim an array of height, width and channels
	 */
	public LFWDataSetIterator(int batchSize, int[] imgDim, boolean useSubset)  {
		super(new LFWLoader(useSubset).getRecordReader(imgDim[0], imgDim[1], imgDim[2]), batchSize, 1, useSubset ? LFWLoader.SUB_NUM_LABELS : LFWLoader.NUM_LABELS);
	}

    /**
     * Create LFW data specific iterator
     * @param batchSize the batch size of the examples
     * @param imgDim an array of height, width and channels
     * @param numExamples the overall number of examples
     * @param useSubset use a subset of the LFWDataSet
     * @param train true if use train value
     * */
    public LFWDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numCategories, boolean useSubset, boolean train, Random rng) {
        super(new LFWLoader(useSubset).getRecordReader(numExamples, batchSize, imgDim[0], imgDim[1], imgDim[2], train, rng), batchSize, 1, numCategories);
    }

	/**
	 * Create LFW data specific iterator
	 * @param batchSize the batch size of the examples
	 * @param imgDim an array of height, width and channels
	 * @param numExamples the overall number of examples
     * @param useSubset use a subset of the LFWDataSet
     * @param train true if use train value
     * @param labelGenerator path label generator to use
	 * */
	public LFWDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numCategories, boolean useSubset, PathLabelGenerator labelGenerator, double splitTrainTest, boolean train, Random rng) {
		super(new LFWLoader(useSubset).getRecordReader(numExamples, batchSize, imgDim[0], imgDim[1], imgDim[2], numCategories, labelGenerator, splitTrainTest, train, rng), batchSize, 1, numCategories);
	}

}
