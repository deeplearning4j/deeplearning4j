package com.ccc.deeplearning.datasets.fetchers;

import org.junit.Test;

import com.ccc.deeplearning.datasets.fetchers.IrisDataFetcher;
import com.ccc.deeplearning.datasets.iterator.DataSetFetcher;

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
