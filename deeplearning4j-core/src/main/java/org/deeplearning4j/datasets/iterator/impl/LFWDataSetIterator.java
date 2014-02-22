package org.deeplearning4j.datasets.iterator.impl;

import org.deeplearning4j.datasets.fetchers.LFWDataFetcher;
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

public class LFWDataSetIterator extends BaseDatasetIterator {

	public LFWDataSetIterator(int batch,int numExamples) {
		super(batch, numExamples,new LFWDataFetcher());
	}

}
