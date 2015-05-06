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

public class LFWDataSetIterator extends BaseDatasetIterator {


	/**
	 * 
	 */
	private static final long serialVersionUID = 7295562586439084858L;
	public LFWDataSetIterator(int batch) {
		this(batch,LFWDataFetcher.NUM_IMAGES,28,28);
	}

	public LFWDataSetIterator(int batch,int imageHeight,int imageWidth) {
		super(batch, LFWDataFetcher.NUM_IMAGES,new LFWDataFetcher(imageWidth,imageHeight));
	}

	public LFWDataSetIterator(int batch,int numExamples) {
		this(batch,numExamples,28,28);
	}
	
	public LFWDataSetIterator(int batch,int numExamples,int imageHeight,int imageWidth) {
		super(batch, numExamples,new LFWDataFetcher(imageWidth,imageHeight));
	}

}
