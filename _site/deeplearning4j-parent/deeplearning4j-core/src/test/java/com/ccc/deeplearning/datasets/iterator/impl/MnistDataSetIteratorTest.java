package com.ccc.deeplearning.datasets.iterator.impl;

import java.io.IOException;

import org.junit.Test;

import com.ccc.deeplearning.datasets.iterator.DataSetIterator;
import com.ccc.deeplearning.datasets.iterator.impl.MnistDataSetIterator;

public class MnistDataSetIteratorTest extends BaseDataSetIteratorTest {

	@Test
	public void testMnist() {
		//everything is 1 based
		this.testNumIters(iter, 6000, false);
		this.testReset(iter, 1);
		this.testCursorPosition(iter, 1, 11);
		this.testInit(iter, 784, 10, 10);
	}


	@Override
	public DataSetIterator getIter() {
		try {
			return new MnistDataSetIterator(10,100);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}


}
