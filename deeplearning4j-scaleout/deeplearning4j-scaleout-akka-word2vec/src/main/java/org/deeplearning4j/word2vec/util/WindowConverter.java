package org.deeplearning4j.word2vec.util;

import java.util.List;

import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.ner.InputHomogenization;
import org.jblas.DoubleMatrix;


/**
 * Util methods for converting windows to 
 * training examples
 * @author Adam Gibson    
 *
 */
public class WindowConverter {
	
	/**
	 * Converts a window (each word in the window)
	 * 
	 * in to a vector.
	 * 
	 * Keep in mind each window is a multi word context.
	 * 
	 * From there, each word uses the passed in model
	 * as a lookup table to get what vectors are relevant
	 * to the passed in windows
	 * @param window the window to take in.
	 * @param vec the model to use as a lookup table
	 * @return a concacneated 1 row array
	 * containing all of the numbers for each word in the window
	 */
	public static double[] asExample(Window window,Word2Vec vec) {
		int length = vec.getLayerSize();
		List<String> words = window.getWords();
		int windowSize = window.getWindowSize();
		
		double[] example = new double[ length * windowSize];
		int count = 0;
		for(int i = 0; i < words.size(); i++) {
			String word = new InputHomogenization(words.get(i)).transform();
			double[] vec2 = vec.getWordVector(word);
			if(vec2 == null)
				vec2 = vec.getOob();
			System.arraycopy(vec2, 0, example, count, length);
			count += length;
		}

		return example;
	}

	

	
	
	public static DoubleMatrix asExampleMatrix(Window window,Word2Vec vec) {
		return new DoubleMatrix(asExample(window,vec));
	}

}
