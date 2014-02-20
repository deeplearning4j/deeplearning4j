package org.deeplearning4j.datasets.iterator;

import java.io.Serializable;

import org.deeplearning4j.datasets.DataSet;


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
