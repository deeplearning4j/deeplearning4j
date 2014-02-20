package org.deeplearning4j.datasets.fetchers;

import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetFetcher;
import org.junit.Test;


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
