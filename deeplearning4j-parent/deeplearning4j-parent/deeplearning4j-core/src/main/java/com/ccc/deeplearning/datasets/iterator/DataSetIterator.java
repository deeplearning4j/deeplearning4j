package com.ccc.deeplearning.datasets.iterator;

import java.io.Serializable;
import java.util.Iterator;

import com.ccc.deeplearning.datasets.DataSet;

public interface DataSetIterator extends Iterator<DataSet>,Serializable {

	
	int totalExamples();
	
	int inputColumns();
	
	int totalOutcomes();
	
	void reset();
	
	int batch();
	
	int cursor();
	
	int numExamples();
	
}
