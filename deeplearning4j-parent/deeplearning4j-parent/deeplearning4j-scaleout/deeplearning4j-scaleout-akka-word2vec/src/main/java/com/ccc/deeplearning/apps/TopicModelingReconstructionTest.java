package com.ccc.deeplearning.apps;

import java.io.File;
import java.io.FileInputStream;

import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.dbn.CDBN;
import com.ccc.deeplearning.topicmodeling.TopicModelingDataSetIterator;

public class TopicModelingReconstructionTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		String dbnLocation = args[0];
		String rootDir = args[1];
		
		
		CDBN c = new CDBN.Builder().buildEmpty();
		c.load(new FileInputStream(new File(dbnLocation)));
		
		int numWords = c.getnIns();
		int numOuts = c.getHiddenLayerSizes()[c.getHiddenLayerSizes().length - 1];
		TopicModelingDataSetIterator iter = new TopicModelingDataSetIterator(new File(rootDir), numOuts, numWords,10);

		
		while(iter.hasNext()) {
			DataSet d = iter.next();
			for(int i = 0; i < d.numExamples(); i++) {
				DataSet d1 = d.get(i);
				int topic = d.get(i).outcome();
				System.out.println("Outcome " + topic + " is " + c.reconstruct(d1.getFirst()));
			}
			
		}

	}

}
