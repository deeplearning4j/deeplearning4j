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
		DataSet filtered = next.filterBy(new int[]{2,4});
		//sets to only 2 labels
		filtered.setNewNumberOfLabels(2);


		for(int i = 0; i < filtered.numExamples(); i++) {
			if(filtered.get(i).outcome() == 2) {
				filtered.setOutcome(i, 0);

			}
			else {
				filtered.setOutcome(i, 1);
			}
		}

		log.info("Number of new examples in data set is " + filtered.numExamples() + " with labels of " + filtered.numOutcomes());
		
		
		BufferedOutputStream fos = new BufferedOutputStream(new FileOutputStream(new File("twoandfours.bin")));
		filtered.write(fos);
		fos.flush();
		fos.close();
		
		
	}

}
