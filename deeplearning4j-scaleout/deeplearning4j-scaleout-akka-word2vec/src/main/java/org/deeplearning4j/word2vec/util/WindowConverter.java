package org.deeplearning4j.word2vec.util;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.inputsanitation.InputHomogenization;


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
	 * as a lookup table to getFromOrigin what vectors are relevant
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
			float[] vec2 = vec.getWordVectorMatrixNormalized(word).floatData();
			if(vec2 == null)
				vec2 = vec.getOob();
            for(int j = 0; j < vec2.length; j++) {
                example[count++] = vec2[j];
            }


		}

		return example;
	}

	

	
	
	public static INDArray asExampleMatrix(Window window,Word2Vec vec) {
		return Nd4j.create(asExample(window, vec));
	}

}
