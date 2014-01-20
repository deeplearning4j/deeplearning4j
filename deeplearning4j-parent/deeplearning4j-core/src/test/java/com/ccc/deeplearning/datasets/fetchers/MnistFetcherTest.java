package com.ccc.deeplearning.datasets.fetchers;

import static org.junit.Assert.*;

import java.io.IOException;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import com.ccc.deeplearning.base.DeepLearningTest;
import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.datasets.fetchers.MnistDataFetcher;
import com.ccc.deeplearning.datasets.iterator.DataSetFetcher;

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
