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

import org.deeplearning4j.datasets.fetchers.LFWDataFetcher;
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

import java.io.IOException;

public class LFWDataSetIterator extends BaseDatasetIterator {

	/**Get the all number of examples for the MNIST training data set.
	 * @param batch the the batch size of the examples
	 */
	public LFWDataSetIterator(int batch) {
		this(batch, LFWDataFetcher.NUM_IMAGES, 250, 250, 3);
	}

	/**Get the all number of examples for the MNIST training data set and set height and width
	 * @param batch the the batch size of the examples
	 * @param imageWidth
	 * @param imageHeight
	 */
	public LFWDataSetIterator(int batch, int imageWidth, int imageHeight)  {
		super(batch, LFWDataFetcher.NUM_IMAGES, new LFWDataFetcher(imageWidth,imageHeight, 3));
	}

	/**Get the specified number of examples for the MNIST training data set.
	 * @param batch the the batch size of the examples
	 * @param numExamples the overall number of examples
	 */
	public LFWDataSetIterator(int batch,int numExamples) {
		this(batch, LFWDataFetcher.NUM_IMAGES, 250, 250, 3);
	}

	/**Get the specified number of examples for the MNIST training data set and set height and width.
	 * @param batch the the batch size of the examples
	 * @param numExamples the overall number of examples
	 * @param imageWidth
	 * @param imageHeight
	 */
	public LFWDataSetIterator(int batch,int numExamples, int imageWidth, int imageHeight, int channels) {
		super(batch, numExamples,new LFWDataFetcher(imageWidth,imageHeight, channels));
	}

	/**Get the specified number of examples for the MNIST training data set and set height and width.
	 * @param batch the the batch size of the examples
	 * @param numExamples the overall number of examples
	 * @param imageWidth
	 * @param imageHeight
	 * @param isSubset use subset sample dataset for A names
	 */
	public LFWDataSetIterator(int batch,int numExamples,int imageWidth, int imageHeight, int channels, boolean isSubset) {
		super(batch, numExamples,new LFWDataFetcher(imageWidth,imageHeight, channels, isSubset));
	}
	/**Get the specified number of examples for the MNIST training data set and set height and width.
	 * @param batch the the batch size of the examples
	 * @param numExamples the overall number of examples
	 * @param imageWidth
	 * @param imageHeight
	 * @param path file path to the subset dataset
	 * @param isSubset use subset sample dataset
	 */
	public LFWDataSetIterator(int batch,int numExamples, int imageWidth, int imageHeight, int channels, String path, boolean isSubset) {
		super(batch, numExamples,new LFWDataFetcher(imageWidth,imageHeight, channels, path, isSubset));
	}

}
