package org.deeplearning4j.datasets.iterator;

import static org.junit.Assert.*;

import java.util.List;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;

public class DataSetIteratorTest {
	
	@Test
	public void testBatchSizeOfOne() throws Exception {
		//Test for (a) iterators returning correct number of examples, and
		//(b) Labels are a proper one-hot vector (i.e., sum is 1.0)
		
		//Iris:
		DataSetIterator iris = new IrisDataSetIterator(1, 5);
		int irisC = 0;
		while(iris.hasNext()){
			irisC++;
			DataSet ds = iris.next();
			assertTrue(ds.getLabels().sum(Integer.MAX_VALUE).getDouble(0)==1.0);
		}
		assertTrue(irisC==5);
		
		
		//MNIST:
		DataSetIterator mnist = new MnistDataSetIterator(1, 5);
		int mnistC = 0;
		while(mnist.hasNext()){
			mnistC++;
			DataSet ds = mnist.next();
			assertTrue(ds.getLabels().sum(Integer.MAX_VALUE).getDouble(0)==1.0);
		}
		assertTrue(mnistC==5);
		
		//LFW:
		DataSetIterator lfw = new LFWDataSetIterator(1, 5);
		int lfwC = 0;
		while(lfw.hasNext()){
			lfwC++;
			DataSet ds = lfw.next();
			assertTrue(ds.getLabels().sum(Integer.MAX_VALUE).getDouble(0)==1.0);
		}
		assertTrue(lfwC==5);
	}

}
