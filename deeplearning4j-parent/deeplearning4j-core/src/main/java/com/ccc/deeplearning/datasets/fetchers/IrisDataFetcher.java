package com.ccc.deeplearning.datasets.fetchers;

import java.io.IOException;

import com.ccc.deeplearning.base.IrisUtils;

public class IrisDataFetcher extends BaseDataFetcher {


	public final static int NUM_EXAMPLES = 150;
	
	public IrisDataFetcher() {
		numOutcomes = 3;
		inputColumns = 4;
		totalExamples = NUM_EXAMPLES;
	}

	@Override
	public void fetch(int numExamples) {
		int from = cursor;
		int to = cursor + numExamples;
		if(to > totalExamples)
			to = totalExamples;
		
		try {
			initializeCurrFromList(IrisUtils.loadIris(from, to));
			cursor += numExamples;
		} catch (IOException e) {
			throw new IllegalStateException("Unable to load iris.dat");
		}
		
	}


}
