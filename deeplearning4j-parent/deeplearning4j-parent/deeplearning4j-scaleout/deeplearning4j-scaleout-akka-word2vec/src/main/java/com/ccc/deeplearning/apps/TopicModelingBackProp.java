package com.ccc.deeplearning.apps;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;

import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.dbn.CDBN;
import com.ccc.deeplearning.topicmodeling.TopicModelingDataSetIterator;

public class TopicModelingBackProp {


	public static void main(String[] args) throws Exception {
		CDBN c = new CDBN.Builder().buildEmpty();
		c.load(new BufferedInputStream(new FileInputStream(new File(args[0]))));
		int numWords = c.getnIns();
		int numOuts = c.getHiddenLayerSizes()[c.getHiddenLayerSizes().length - 1];
		TopicModelingDataSetIterator iter = new TopicModelingDataSetIterator(new File(args[1]), numOuts, numWords,10);


		while(iter.hasNext()) {
			DataSet d = iter.next();
			c.setInput(d.getFirst());
			c.backProp(0.01, 1000);

		}
		
		
		BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(args[2]));
		c.write(bos);
		bos.flush();
		bos.close();


	}

}
