package org.deeplearning4j.eval;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import org.deeplearning4j.base.DeepLearningTest;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


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
