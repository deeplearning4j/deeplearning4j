package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base;

import java.io.IOException;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;



public abstract class DeepLearningTest {
	
	private static Logger log = LoggerFactory.getLogger(DeepLearningTest.class);
	
	public Pair<DoubleMatrix,DoubleMatrix> getIris() throws IOException {
		Pair<DoubleMatrix,DoubleMatrix> pair = IrisUtils.loadIris();
		return pair;
	}
	

	

	

}
