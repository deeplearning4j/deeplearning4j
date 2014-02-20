package org.deeplearning4j.datasets.iterator;

import java.io.Serializable;
import java.util.Iterator;

import org.deeplearning4j.datasets.DataSet;


public interface DataSetIterator extends Iterator<DataSet>,Serializable {

	
	int totalExamples();
	
	int inputColumns();
	
	int totalOutcomes();
	
	void reset();
	
	int batch();
	
	int cursor();
	
	int numExamples();
	
}
