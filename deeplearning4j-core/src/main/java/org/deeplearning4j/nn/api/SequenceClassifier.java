package org.deeplearning4j.nn.api;

import org.deeplearning4j.linalg.api.ndarray.INDArray;

/**
 * Created by agibsonccc on 8/27/14.
 */
public interface SequenceClassifier {


    Classifier classifier();


    int mostLikelyInSequence(INDArray examples);


    INDArray predict(INDArray examples);

    void fit(INDArray features,INDArray labels);



}
