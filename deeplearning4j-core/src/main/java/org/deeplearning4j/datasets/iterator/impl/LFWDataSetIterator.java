package org.deeplearning4j.datasets.iterator.impl;

import org.deeplearning4j.datasets.fetchers.LFWDataFetcher;
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

public class LFWDataSetIterator extends BaseDatasetIterator {


	/**
	 * 
	 */
	private static final long serialVersionUID = 7295562586439084858L;

	public LFWDataSetIterator(int batch,int numExamples) {
		this(batch,numExamples,28,28);
	}
	
	public LFWDataSetIterator(int batch,int numExamples,int imageHeight,int imageWidth) {
		super(batch, numExamples,new LFWDataFetcher(imageWidth,imageHeight));
	}

}
