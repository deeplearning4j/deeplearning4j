package org.deeplearning4j.zoo.util;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * Interface to helper classes that return label descriptions.
 *
 * @author saudet
 */
public interface Labels {

    /**
     * Returns the description of the nth class from the classes of a dataset.
     * @param n
     * @return label description
     */
    String getLabel(int n);

    /**
     * Given predictions from the trained model this method will return a list
     * of the top n matches and the respective probabilities.
     * @param predictions raw
     * @return decoded predictions
     */
    List<List<ClassPrediction>> decodePredictions(INDArray predictions, int n);
}
