package org.deeplearning4j.example.mnist;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.RawMnistDataSetIterator;
import org.deeplearning4j.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Create2sAnd4sDataSet {

	
	private static Logger log = LoggerFactory.getLogger(Create2sAnd4sDataSet.class);
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		DataSetIterator iter = new RawMnistDataSetIterator(60000,60000);
		DataSet next = iter.next();
		next.filterAndStrip(new int[]{2,4});
		log.info("Number of new examples in data set is " + next.numExamples() + " with labels of " + next.numOutcomes());
		
		
		BufferedOutputStream fos = new BufferedOutputStream(new FileOutputStream(new File("twoandfours.bin")));
		next.write(fos);
		fos.flush();
		fos.close();
		
		
	}

}
