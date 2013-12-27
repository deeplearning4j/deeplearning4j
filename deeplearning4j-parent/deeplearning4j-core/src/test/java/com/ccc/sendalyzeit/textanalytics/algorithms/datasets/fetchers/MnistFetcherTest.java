package com.ccc.sendalyzeit.textanalytics.algorithms.datasets.fetchers;

import static org.junit.Assert.*;

import java.io.IOException;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator.DataSetFetcher;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;

public class MnistFetcherTest extends BaseDataFetcherTest {

	
	
	@Test
	public void testMnistFetcher() throws IOException {
		Pair<DoubleMatrix,DoubleMatrix> pair = DeepLearningTest.getMnistExample(1);
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
