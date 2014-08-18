package org.deeplearning4j.datasets.iterator;

import static org.junit.Assert.*;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.jblas.NDArray;
import org.jblas.DoubleMatrix;
import org.junit.Test;

public class ListDataSetIteratorTest {

	@Test
	public void testIter() {
		NDArray x = new NDArray(new DoubleMatrix(new double[][] {
				{1,2,3,4,5,6},
				{1,2,3,4,5,6},
				{1,2,3,4,5,6},
				{1,2,3,4,5,6},
				{1,2,3,4,5,6},

		}));
		
		NDArray y  = new NDArray(new DoubleMatrix(new double[][]{
				{1,0,0},
				{0,1,0},
				{0,0,1},
				{1,0,0},
				{0,1,0}
		}));
		
		
		DataSet d = new DataSet(x,y);
		
		assertEquals(true,d.numExamples() == 5);
		
		DataSetIterator iter = new ListDataSetIterator(d.dataSetBatches(2));
		assertEquals(true,iter.hasNext());
		
		assertEquals(iter.numExamples(),3);
		
		DataSet next = iter.next();
		//odd number of items
		assertEquals(5,next.numExamples());
		
		for(int i = 0; i < next.numExamples(); i++) {
			DataSet d1 = next.get(i);
			assertEquals(d1.getFeatureMatrix(),x.getRow(i));
		}
		
	}

}
