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

import org.canova.image.loader.LFWLoader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;

import java.util.Random;


public class LFWDataSetIterator extends RecordReaderDataSetIterator {

	protected static int width = 250;
	protected static int height = 250;
	protected static int channels = 3;

	/**
	 * Create LFW data specific iterator
	 * @param batchSize the the batch size of the examples
	 * @param numExamples the overall number of examples
	 * */
	public LFWDataSetIterator(int batchSize, int numExamples) {
		super(new LFWLoader().getRecordReader(numExamples), batchSize, width * height * channels, LFWLoader.NUM_LABELS);
	}

	/**
	 * Create LFW data specific iterator
	 * @param batchSize the the batch size of the examples
	 * @param imgDim an array of width, height and channels
	 * @param numExamples the overall number of examples
	 * */
	public LFWDataSetIterator(int batchSize, int numExamples, int[] imgDim) {
		super(new LFWLoader().getRecordReader(imgDim[0], imgDim[1], imgDim[2], numExamples), batchSize, imgDim[0] * imgDim[1] * imgDim[2], LFWLoader.NUM_LABELS);
	}

	/**
	 * Create LFW data specific iterator
	 * @param batchSize the the batch size of the examples
	 * @param imgDim an array of width, height and channels
	 * @param numExamples the overall number of examples
	 * */
	public LFWDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numCategories) {
		super(new LFWLoader().getRecordReader(imgDim[0], imgDim[1], imgDim[2], numExamples), batchSize, imgDim[0] * imgDim[1] * imgDim[2], numCategories);
	}

	/**
	 * Create LFW data specific iterator
	 * @param batchSize the the batch size of the examples
	 * @param numExamples the overall number of examples
	 * @param numCategories the overall number of labels
	 * */
	public LFWDataSetIterator(int batchSize, int numExamples, int numCategories) {
		super(new LFWLoader().getRecordReader(numExamples, numCategories), batchSize, width * height * channels, numCategories);
	}

	/**
	 * Create LFW data specific iterator
	 * @param batchSize the the batch size of the examples
	 * @param imgDim an array of width, height and channels
	 */
	public LFWDataSetIterator(int batchSize, int[] imgDim)  {
		super(new LFWLoader().getRecordReader(imgDim[0], imgDim[1], imgDim[2]), batchSize, imgDim[0] * imgDim[1] * imgDim[2], LFWLoader.NUM_LABELS);
	}


	/**
	 * Create LFW data specific iterator
	 * @param batchSize the the batch size of the examples
	 * @param numExamples the overall number of examples
	 * @param numCategories the overall number of labels
	 * */
	public LFWDataSetIterator(int batchSize, int numExamples, int numCategories, boolean useSubset) {
		super(new LFWLoader(useSubset).getRecordReader(numExamples, numCategories), batchSize, width * height * channels, numCategories);
	}

	/**
	 * Create LFW data specific iterator
	 * @param batchSize the the batch size of the examples
	 * @param imgDim an array of width, height and channels
	 */
	public LFWDataSetIterator(int batchSize, int[] imgDim, boolean useSubset)  {
		super(new LFWLoader(useSubset).getRecordReader(imgDim[0], imgDim[1], imgDim[2]), batchSize, imgDim[0] * imgDim[1] * imgDim[2], useSubset ? LFWLoader.SUB_NUM_LABELS : LFWLoader.NUM_LABELS);
	}


	/**
	 * Create LFW data specific iterator
	 * @param batchSize the the batch size of the examples
	 * @param imgDim an array of width, height and channels
	 * @param numExamples the overall number of examples
	 * */
	public LFWDataSetIterator(int batchSize, int numExamples, int[] imgDim, int numCategories, boolean useSubset, Random rng) {
		super(new LFWLoader(useSubset).getRecordReader(imgDim[0], imgDim[1], imgDim[2], numExamples, numCategories, rng), batchSize, imgDim[0] * imgDim[1] * imgDim[2], numCategories);
	}

}
