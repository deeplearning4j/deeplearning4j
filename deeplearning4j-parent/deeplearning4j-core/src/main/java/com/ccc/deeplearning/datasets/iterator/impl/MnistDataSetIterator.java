package com.ccc.deeplearning.datasets.iterator.impl;

import java.io.IOException;

import com.ccc.deeplearning.datasets.fetchers.MnistDataFetcher;
import com.ccc.deeplearning.datasets.iterator.BaseDatasetIterator;

public class MnistDataSetIterator extends BaseDatasetIterator {

	public MnistDataSetIterator(int batch,int numExamples) throws IOException {
		super(batch, numExamples,new MnistDataFetcher());
	}


}
