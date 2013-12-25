package com.ccc.sendalyzeit.deeplearning.eval;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;

public class ModelTester {

	
	private static Logger log = LoggerFactory.getLogger(ModelTester.class);
	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		Pair<DoubleMatrix,DoubleMatrix> inputs = DeepLearningTest.getMnistExampleBatch(100);
		Evaluation eval = new Evaluation();
		BaseMultiLayerNetwork load = BaseMultiLayerNetwork.loadFromFile(new FileInputStream(new File(args[0])));
		DoubleMatrix in = inputs.getFirst();
		DoubleMatrix outcomes = inputs.getSecond();

		for(int i = 0; i < in.rows; i++) {
			DoubleMatrix pre = load.predict(in.getRow(i));
			eval.eval(outcomes.getRow(i), pre);
		}
		
		
		log.info(eval.stats());
	}

}
