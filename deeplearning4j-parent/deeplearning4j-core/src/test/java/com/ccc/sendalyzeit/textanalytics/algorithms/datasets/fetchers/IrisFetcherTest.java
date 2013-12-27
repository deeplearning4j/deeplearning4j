package com.ccc.sendalyzeit.textanalytics.algorithms.datasets.fetchers;

import org.junit.Test;

import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator.DataSetFetcher;

public class IrisFetcherTest extends BaseDataFetcherTest {

	@Override
	public DataSetFetcher getFetcher() {
	    return new IrisDataFetcher();
	}

	@Test
	public void testIrisFetcher() {
		testFetcher(fetcher, 4, 3);
		this.testFetchBatchSize(10);
	}
	
	
}
