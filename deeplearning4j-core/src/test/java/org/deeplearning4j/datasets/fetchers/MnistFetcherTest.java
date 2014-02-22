package org.deeplearning4j.datasets.fetchers;

import static org.junit.Assert.*;

import java.io.IOException;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetFetcher;
import org.jblas.DoubleMatrix;
import org.junit.Test;


public class MnistFetcherTest extends BaseDataFetcherTest {

	
	
	@Test
	public void testMnistFetcher() throws IOException {
		DataSetFetcher fetcher = getFetcher();
		fetcher.fetch(10);
		Pair<DoubleMatrix,DoubleMatrix> pair = fetcher.next();
		int inputColumns = pair.getFirst().columns;
		int outputColumns = 10;
		testFetcher(fetcher, inputColumns, outputColumns);
		testFetchBatchSize(10);
		assertEquals(true,fetcher.hasMore());
	}
	
	@Override
	public DataSetFetcher getFetcher() {
		try {
			return new MnistDataFetcher();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	

}
