/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.text.movingwindow;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.models.word2vec.Word2Vec;
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
        int length = vec.lookupTable().layerSize();
        List<String> words = window.getWords();
        int windowSize = vec.getWindow();
        assert words.size() == vec.getWindow();
        INDArray ret = Nd4j.create(length * windowSize);



        for(int i = 0; i < words.size(); i++) {
            String word = words.get(i);
            INDArray n = normalize ? vec.getWordVectorMatrixNormalized(word) :  vec.getWordVectorMatrix(word);
            ret.put(new NDArrayIndex[]{NDArrayIndex.interval(i * vec.lookupTable().layerSize(),i * vec.lookupTable().layerSize() + vec.lookupTable().layerSize())},n);
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
		int length = vec.lookupTable().layerSize();
		List<String> words = window.getWords();
		int windowSize = window.getWindowSize();
		
		double[] example = new double[ length * windowSize];
		int count = 0;
		for(int i = 0; i < words.size(); i++) {
			String word = words.get(i);
            INDArray n = vec.getWordVectorMatrixNormalized(word);
			double[] vec2 = n == null ? vec.getWordVectorMatrix(Word2Vec.UNK).data().asDouble() : vec.getWordVectorMatrix(word).data().asDouble();
			if(vec2 == null)
				vec2 = vec.getWordVectorMatrix(Word2Vec.UNK).data().asDouble();
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
