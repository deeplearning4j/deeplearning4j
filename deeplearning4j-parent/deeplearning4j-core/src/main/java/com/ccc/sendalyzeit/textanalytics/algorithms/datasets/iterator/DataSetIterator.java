package com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator;

import java.util.Iterator;

import org.jblas.DoubleMatrix;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;

public interface DataSetIterator extends Iterator<Pair<DoubleMatrix,DoubleMatrix>>{

	
	int totalExamples();
	
	int inputColumns();
	
	int totalOutcomes();
	
	void reset();
	
}
