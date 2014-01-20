package com.ccc.deeplearning.datasets.iterator.impl;

import com.ccc.deeplearning.datasets.fetchers.IrisDataFetcher;
import com.ccc.deeplearning.datasets.iterator.BaseDatasetIterator;

public class IrisDataSetIterator extends BaseDatasetIterator {

	public IrisDataSetIterator(int batch,int numExamples) {
		super(batch,numExamples,new IrisDataFetcher());
	}

	

}
