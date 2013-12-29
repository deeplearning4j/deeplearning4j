package com.ccc.deeplearning.datasets.iterator.impl;

import com.ccc.deeplearning.datasets.fetchers.LFWDataFetcher;
import com.ccc.deeplearning.datasets.iterator.BaseDatasetIterator;

public class LFWDataSetIterator extends BaseDatasetIterator {

	public LFWDataSetIterator(int batch,int numExamples) {
		super(batch, numExamples,new LFWDataFetcher());
	}

}
