package com.ccc.deeplearning.datasets.iterator;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.datasets.DataSet;

public interface DataSetFetcher extends Serializable {

	
	boolean hasMore();
	
	DataSet next();
	
	void fetch(int numExamples);
	
	int totalOutcomes();
	
	int inputColumns();
	
	int totalExamples();
	
	void reset();
	
	int cursor();
}
