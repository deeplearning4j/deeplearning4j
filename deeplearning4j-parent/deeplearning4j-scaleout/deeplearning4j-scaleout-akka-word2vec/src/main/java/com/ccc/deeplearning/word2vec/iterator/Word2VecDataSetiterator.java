package com.ccc.deeplearning.word2vec.iterator;

import java.io.Serializable;
import java.util.Iterator;

import com.ccc.deeplearning.word2vec.dataset.Word2VecDataSet;

public interface Word2VecDataSetiterator  extends Iterator<Word2VecDataSet>,Serializable {

		
		int totalExamples();
		
		int inputColumns();
		
		int totalOutcomes();
		
		void reset();
		
		int batch();
		
		int cursor();
		
		int numExamples();
		
	
}
