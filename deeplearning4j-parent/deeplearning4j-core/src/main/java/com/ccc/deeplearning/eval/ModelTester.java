package com.ccc.deeplearning.eval;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.base.DeepLearningTest;
import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.datasets.fetchers.MnistDataFetcher;
import com.ccc.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import com.ccc.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;

public class ModelTester {

	
	private static Logger log = LoggerFactory.getLogger(ModelTester.class);
	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		MnistDataSetIterator iter = new MnistDataSetIterator(10, 60000);
		
		Evaluation eval = new Evaluation();
		BaseMultiLayerNetwork load = BaseMultiLayerNetwork.loadFromFile(new FileInputStream(new File(args[0])));
		while(iter.hasNext()) {
			Pair<DoubleMatrix,DoubleMatrix> inputs = iter.next();

			DoubleMatrix in = inputs.getFirst();
			DoubleMatrix outcomes = inputs.getSecond();
			DoubleMatrix predicted = load.predict(in);
			eval.eval(outcomes, predicted);
		}
		
		
		
		
		log.info(eval.stats());
	}

}
