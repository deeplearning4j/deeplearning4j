package org.deeplearning4j.text.movingwindow;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.inputsanitation.InputHomogenization;
import org.nd4j.linalg.indexing.NDArrayIndex;


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
    public static INDArray asExampleArray(Window window,Word2Vec vec,boolean normalize) {
        int length = vec.getLayerSize();
        List<String> words = window.getWords();
        int windowSize = vec.getWindow();
        assert words.size() == vec.getWindow();
        INDArray ret = Nd4j.create(length * windowSize);



        for(int i = 0; i < words.size(); i++) {
            String word = words.get(i);
            INDArray n = normalize ? vec.getWordVectorMatrixNormalized(word) :  vec.getWordVectorMatrix(word);
            ret.put(new NDArrayIndex[]{NDArrayIndex.interval(i * vec.getLayerSize(),i * vec.getLayerSize() + vec.getLayerSize())},n);
        }

        return ret;
    }


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
			String word = words.get(i);
            INDArray n = vec.getWordVectorMatrixNormalized(word);
			float[] vec2 = n == null ? vec.getWordVectorMatrix(Word2Vec.UNK).data() : vec.getWordVectorMatrix(word).data();
			if(vec2 == null)
				vec2 = vec.getWordVectorMatrix(Word2Vec.UNK).data();
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
