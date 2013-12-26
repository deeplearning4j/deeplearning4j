package com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;

public interface DataSetFetcher extends Serializable {

	
	boolean hasMore();
	
	Pair<DoubleMatrix,DoubleMatrix> next();
	
	void fetch(int numExamples);
	
	int totalOutcomes();
	
	int inputColumns();
	
	int totalExamples();
	
	void reset();
	
	int cursor();
}
