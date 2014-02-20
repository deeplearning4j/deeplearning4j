package org.deeplearning4j.word2vec.iterator;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;

import org.deeplearning4j.word2vec.util.Window;


public interface Word2VecDataSetIterator  extends Iterator<List<Window>>,Serializable {

		
		int totalExamples();
		
		int inputColumns();
		
		int totalOutcomes();
		
		void reset();
		
		int batch();
		
		int cursor();
		
		int numExamples();
		
	
}
