package org.deeplearning4j.models.embeddings;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.plot.Tsne;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Iterator;
import java.util.concurrent.atomic.AtomicLong;

/**
 * General weight lookup table
 *
 * @author Adam Gibson
 */
public interface WeightLookupTable extends Serializable {





    /**
     * Render the words via TSNE
     * @param tsne the tsne to use
     */
    void plotVocab(Tsne tsne);

    /**
     * Render the words via tsne
     */
    void plotVocab();

    /**
     *
     * @param codeIndex
     * @param code
     */
    void putCode(int codeIndex,INDArray code);

    /**
     * Loads the co-occurrences for the given codes
     * @param codes the codes to load
     * @return an ndarray of code.length by layerSize
     */
    INDArray loadCodes(int[] codes);

    /**
     * Iterate on the given 2 vocab words
     * @param w1 the first word to iterate on
     * @param w2 the second word to iterate on
     */
    void iterate(VocabWord w1,VocabWord w2);
    /**
     * Iterate on the given 2 vocab words
     * @param w1 the first word to iterate on
     * @param w2 the second word to iterate on
     * @param nextRandom nextRandom for sampling
     * @param alpha the alpha to use for learning
     */
    void iterateSample(VocabWord w1,VocabWord w2,AtomicLong nextRandom,double alpha);


    /**
     * Inserts a word vector
     * @param word the word to insert
     * @param vector the vector to insert
     */
    void putVector(String word,INDArray vector);

    /**
     *
     * @param word
     * @return
     */
    INDArray vector(String word);

    /**
     * Reset the weights of the cache
     */
    void resetWeights();


    /**
     * Sets the learning rate
     * @param lr
     */
    void setLearningRate(double lr);

    /**
     * Iterates through all of the vectors in the cache
     * @return an iterator for all vectors in the cache
     */
    Iterator<INDArray> vectors();

}
