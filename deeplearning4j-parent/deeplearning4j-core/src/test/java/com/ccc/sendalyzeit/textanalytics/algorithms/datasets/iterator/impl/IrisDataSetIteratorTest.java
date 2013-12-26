package com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator.impl;

import org.junit.Test;

import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator.DataSetIterator;

public class IrisDataSetIteratorTest extends BaseDataSetIteratorTest {

	
	@Test
	public void testIris() {
		this.testCursorPosition(iter, 1, 10);
		this.testInit(iter, 4, 3, 10);
		this.testReset(iter, 0);
		this.testNumIters(iter, 1, true);
	}
	
	
	
	
	@Override
	public DataSetIterator getIter() {
		return new IrisDataSetIterator(10,100);
	}


}
