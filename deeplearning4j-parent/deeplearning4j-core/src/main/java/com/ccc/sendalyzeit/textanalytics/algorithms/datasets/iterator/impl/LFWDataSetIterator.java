package com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator.impl;

import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.fetchers.LFWDataFetcher;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator.BaseDatasetIterator;

public class LFWDataSetIterator extends BaseDatasetIterator {

	public LFWDataSetIterator(int batch) {
		super(batch, new LFWDataFetcher());
	}

}
