package org.deeplearning4j.datasets.iterator;

import static org.junit.Assert.*;

import java.io.IOException;
import java.util.List;

import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.springframework.core.io.ClassPathResource;

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

	@Test
	public void testMnist() throws Exception {
		ClassPathResource cpr = new ClassPathResource("mnist_first_200.txt");
		CSVRecordReader rr = new CSVRecordReader(0,",");
		rr.initialize(new FileSplit(cpr.getFile()));
		RecordReaderDataSetIterator dsi = new RecordReaderDataSetIterator(rr,0,10);

		MnistDataSetIterator iter = new MnistDataSetIterator(10,200,false,true,false,0);

		while(dsi.hasNext()){
			DataSet dsExp = dsi.next();
			DataSet dsAct = iter.next();

			INDArray fExp = dsExp.getFeatureMatrix();
			fExp.divi(255);
			INDArray lExp = dsExp.getLabels();

			INDArray fAct = dsAct.getFeatureMatrix();
			INDArray lAct = dsAct.getLabels();

			assertEquals(fExp,fAct);
			assertEquals(lExp,lAct);
		}
		assertFalse(iter.hasNext());
	}

}
