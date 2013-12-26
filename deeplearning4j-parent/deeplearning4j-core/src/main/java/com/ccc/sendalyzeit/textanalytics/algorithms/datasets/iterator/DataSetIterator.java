package com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator;

import java.io.Serializable;
import java.util.Iterator;

import org.jblas.DoubleMatrix;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;

public interface DataSetIterator extends Iterator<Pair<DoubleMatrix,DoubleMatrix>>,Serializable {

	
	int totalExamples();
	
	int inputColumns();
	
	int totalOutcomes();
	
	void reset();
	
	int batch();
	
	int cursor();
	
	int numExamples();
	
}
