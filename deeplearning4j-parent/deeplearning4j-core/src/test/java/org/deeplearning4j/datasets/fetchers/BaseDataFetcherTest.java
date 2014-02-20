package org.deeplearning4j.datasets.fetchers;

import static org.junit.Assert.*;
import static org.junit.Assume.*;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.DataSetFetcher;
import org.jblas.DoubleMatrix;
import org.junit.Before;
import org.junit.Test;



public abstract class BaseDataFetcherTest {

	protected DataSetFetcher fetcher;
	
	@Before
	public void setup() {
		fetcher = getFetcher();
		assumeNotNull(fetcher);
	}
	
	
	public void testFetcher(DataSetFetcher fetcher,int inputColumnsExpected,int totalOutcomesExpected) {
		assertEquals(inputColumnsExpected,fetcher.inputColumns());
		assertEquals(totalOutcomesExpected,fetcher.totalOutcomes());
		assertEquals(true,fetcher.hasMore());
		assertEquals(true,fetcher.next() != null && fetcher.hasMore());
		
	}
	
	public void testFetchBatchSize(int expectedBatchSize) {
		fetcher.fetch(expectedBatchSize);
		assumeNotNull(fetcher.next());
		assertEquals(expectedBatchSize,fetcher.next().getFirst().rows);
		
	}
	

	
	
	
	public abstract DataSetFetcher getFetcher();

}
