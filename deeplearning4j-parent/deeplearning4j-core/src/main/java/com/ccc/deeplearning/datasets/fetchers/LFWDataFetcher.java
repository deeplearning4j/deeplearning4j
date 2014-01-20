package com.ccc.deeplearning.datasets.fetchers;

import java.util.ArrayList;
import java.util.List;

import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.base.LFWLoader;
import com.ccc.deeplearning.berkeley.Pair;

/**
 * Data fetcher for the LFW faces dataset
 * @author Adam Gibson
 *
 */
public class LFWDataFetcher extends BaseDataFetcher {

	private LFWLoader loader = new LFWLoader();
	public final static int NUM_IMAGES = 13233;



	public LFWDataFetcher() {
		try {
			loader.getIfNotExists();
			inputColumns = loader.getNumPixelColumns();
			numOutcomes = loader.getNumNames();
			totalExamples = NUM_IMAGES;
		} catch (Exception e) {
			throw new IllegalStateException("Unable to fetch images");
		}
	}




	@Override
	public void fetch(int numExamples) {
		if(!hasMore())
			throw new IllegalStateException("Unable to get more; there are no more images");



		//we need to ensure that we don't overshoot the number of examples total
		List<Pair<DoubleMatrix,DoubleMatrix>> toConvert = new ArrayList<>();

		for(int i = 0; i < numExamples; i++,cursor++) {
			if(!hasMore())
				break;
			toConvert.add(loader.getDataFor(cursor));
		}

		initializeCurrFromList(toConvert);
	}



}
