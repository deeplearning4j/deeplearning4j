package org.deeplearning4j.models.oneHotVector.api;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Collection;

public interface WordVectorsAPI extends Serializable {
    String getUNK();
    void setUNK(String newUNK);

    /**
     * Returns true if the model has this word in the vocab
     * @param word the word to test for
     * @return true if the model has the word in the vocab
     */
    boolean hasWord(String word);

    /**
     * Get the word vector for a given matrix
     * @param word the word to get the matrix for
     * @return the ndarray for this word
     */
    INDArray getWordVectorMatrix(String word);

    /**
     * This method returns 2D array, where each row represents corresponding word/label
     *
     * @param labels
     * @return
     */
    INDArray getWordVectors(Collection<String> labels);

    /**
     * @return model length (features for each vector)
     */
    int getLength();
}
