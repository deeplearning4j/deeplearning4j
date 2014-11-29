package org.deeplearning4j.scaleout.perform.models.word2vec;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by agibsonccc on 11/29/14.
 */
public class Word2VecWork implements Serializable {

    private Map<String,Pair<VocabWord,INDArray>> vectors = new HashMap<>();





}
