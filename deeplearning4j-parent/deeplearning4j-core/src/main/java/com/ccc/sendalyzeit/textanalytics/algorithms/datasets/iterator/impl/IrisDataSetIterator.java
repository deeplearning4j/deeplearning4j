package com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator.impl;

import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.fetchers.IrisDataFetcher;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator.BaseDatasetIterator;

public class IrisDataSetIterator extends BaseDatasetIterator {

	public IrisDataSetIterator(int batch) {
		super(batch, new IrisDataFetcher());
	}

	

}
