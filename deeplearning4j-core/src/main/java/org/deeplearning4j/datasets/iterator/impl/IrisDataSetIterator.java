package org.deeplearning4j.datasets.iterator.impl;

import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

public class IrisDataSetIterator extends BaseDatasetIterator {

	public IrisDataSetIterator(int batch,int numExamples) {
		super(batch,numExamples,new IrisDataFetcher());
	}

	

}
