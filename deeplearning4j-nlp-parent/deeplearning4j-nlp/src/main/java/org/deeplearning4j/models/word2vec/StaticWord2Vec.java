package org.deeplearning4j.models.word2vec;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * @author raver119@gmail.com
 */
public class StaticWord2Vec implements WordVectors {
    @Override
    public String getUNK() {
        return null;
    }

    @Override
    public void setUNK(String newUNK) {

    }

    /**
     * Returns true if the model has this word in the vocab
     *
     * @param word the word to test for
     * @return true if the model has the word in the vocab
     */
    @Override
    public boolean hasWord(String word) {
        return false;
    }

    @Override
    public Collection<String> wordsNearest(INDArray words, int top) {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    @Override
    public Collection<String> wordsNearestSum(INDArray words, int top) {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    /**
     * Get the top n words most similar to the given word
     * PLEASE NOTE: This method is not available in this implementation.
     *
     * @param word the word to compare
     * @param n    the n to get
     * @return the top n words
     */
    @Override
    public Collection<String> wordsNearestSum(String word, int n) {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    /**
     * Words nearest based on positive and negative words
     * PLEASE NOTE: This method is not available in this implementation.
     *
     * @param positive the positive words
     * @param negative the negative words
     * @param top      the top n words
     * @return the words nearest the mean of the words
     */
    @Override
    public Collection<String> wordsNearestSum(Collection<String> positive, Collection<String> negative, int top) {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    /**
     * Accuracy based on questions which are a space separated list of strings
     * where the first word is the query word, the next 2 words are negative,
     * and the last word is the predicted word to be nearest
     * PLEASE NOTE: This method is not available in this implementation.
     *
     * @param questions the questions to ask
     * @return the accuracy based on these questions
     */
    @Override
    public Map<String, Double> accuracy(List<String> questions) {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    @Override
    public int indexOf(String word) {
        return 0;
    }

    /**
     * Find all words with a similar characters
     * in the vocab
     * PLEASE NOTE: This method is not available in this implementation.
     *
     * @param word     the word to compare
     * @param accuracy the accuracy: 0 to 1
     * @return the list of words that are similar in the vocab
     */
    @Override
    public List<String> similarWordsInVocabTo(String word, double accuracy) {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    /**
     * Get the word vector for a given matrix
     *
     * @param word the word to get the matrix for
     * @return the ndarray for this word
     */
    @Override
    public double[] getWordVector(String word) {
        return new double[0];
    }

    /**
     * Returns the word vector divided by the norm2 of the array
     *
     * @param word the word to get the matrix for
     * @return the looked up matrix
     */
    @Override
    public INDArray getWordVectorMatrixNormalized(String word) {
        return null;
    }

    /**
     * Get the word vector for a given matrix
     *
     * @param word the word to get the matrix for
     * @return the ndarray for this word
     */
    @Override
    public INDArray getWordVectorMatrix(String word) {
        return null;
    }

    /**
     * This method returns 2D array, where each row represents corresponding word/label
     *
     * @param labels
     * @return
     */
    @Override
    public INDArray getWordVectors(Collection<String> labels) {
        return null;
    }

    /**
     * This method returns mean vector, built from words/labels passed in
     *
     * @param labels
     * @return
     */
    @Override
    public INDArray getWordVectorsMean(Collection<String> labels) {
        return null;
    }

    /**
     * Words nearest based on positive and negative words
     * PLEASE NOTE: This method is not available in this implementation.
     *
     * @param positive the positive words
     * @param negative the negative words
     * @param top      the top n words
     * @return the words nearest the mean of the words
     */
    @Override
    public Collection<String> wordsNearest(Collection<String> positive, Collection<String> negative, int top) {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    /**
     * Get the top n words most similar to the given word
     * PLEASE NOTE: This method is not available in this implementation.
     *
     * @param word the word to compare
     * @param n    the n to get
     * @return the top n words
     */
    @Override
    public Collection<String> wordsNearest(String word, int n) {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    /**
     * Returns the similarity of 2 words
     *
     * @param word  the first word
     * @param word2 the second word
     * @return a normalized similarity (cosine similarity)
     */
    @Override
    public double similarity(String word, String word2) {
        return 0;
    }

    /**
     * Vocab for the vectors
     *
     * @return
     */
    @Override
    public VocabCache vocab() {
        return null;
    }

    /**
     * Lookup table for the vectors
     * PLEASE NOTE: This method is not available in this implementation.
     *
     * @return
     */
    @Override
    public WeightLookupTable lookupTable() {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    /**
     * Specifies ModelUtils to be used to access model
     * PLEASE NOTE: This method has no effect in this implementation.
     *
     * @param utils
     */
    @Override
    public void setModelUtils(ModelUtils utils) {

    }
}
