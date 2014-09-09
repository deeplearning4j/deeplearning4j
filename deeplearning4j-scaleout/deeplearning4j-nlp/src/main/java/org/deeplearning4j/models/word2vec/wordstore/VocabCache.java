package org.deeplearning4j.models.word2vec.wordstore;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.models.word2vec.VocabWord;

import java.util.Collection;

/**
 * A VocabCache handles the storage of information needed for the word2vec look up table.
 *
 * @author Adam Gibson
 */
public interface VocabCache  {


    /**
     * Increment the count for the given word
     * @param word the word to increment the count for
     */
    void incrementWordCount(String word);


    /**
     * Increment the count for the given word by
     * the amount increment
     * @param word the word to increment the count for
     * @param increment the amount to increment by
     */
    void incrementWordCount(String word,int increment);

    /**
     * Returns the number of times the word has occurred
     * @param word the word to retrieve the occurrence frequency for
     * @return 0 if hasn't occurred or the number of times
     * the word occurs
     */
    int wordFrequency(String word);

    /**
     * Returns true if the cache contains the given word
     * @param word the word to check for
     * @return
     */
    boolean containsWord(String word);

    /**
     * Returns the word contained at the given index or null
     * @param index the index of the word to get
     * @return the word at the given index
     */
    String wordAtIndex(int index);

    /**
     * Returns the index of a given word
     * @param word the index of a given word
     * @return the index of a given word or -1
     * if not found
     */
    int indexOf(String word);


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
     * Returns all of the vocab word nodes
     * @return
     */
    Collection<VocabWord> vocabWords();


    /**
     * The total number of word occurrences
     * @return the total number of word occurrences
     */
    int totalWordOccurrences();

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
     *
     * @param word
     * @return
     */
    VocabWord wordFor(String word);


    /**
     *
     * @param index
     * @param word
     */
    void addWordToIndex(int index,String word);

    /**
     *
     * @param word
     * @param vocabWord
     */
    void putVocabWord(String word,VocabWord vocabWord);

    /**
     * Returns the number of words in the cache
     * @return the number of words in the cahce
     */
    int numWords();

}
