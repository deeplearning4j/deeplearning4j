package com.ccc.deeplearning.datasets.iterator.impl;

import org.junit.Test;

import com.ccc.deeplearning.datasets.iterator.DataSetIterator;
import com.ccc.deeplearning.datasets.iterator.impl.IrisDataSetIterator;

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
