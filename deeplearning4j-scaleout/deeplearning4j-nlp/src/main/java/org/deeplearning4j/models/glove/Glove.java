package org.deeplearning4j.models.glove;

import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import java.io.Serializable;

/**
 * Glove by socher et. al
 *
 * @author Adam Gibson
 */
public class Glove implements Serializable {

    private VocabCache vocabCache;
    private TextVectorizer textVectorizer;
    private WeightLookupTable weightLookupTable;
    private int layerSize = 100;
    private double learningRate = 0.1;
    private double xMax = 0.75;


}
