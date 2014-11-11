package org.deeplearning4j.models.word2vec.wordstore;

import org.deeplearning4j.plot.Tsne;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.models.word2vec.VocabWord;

import java.util.Collection;
import java.util.Iterator;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A VocabCache handles the storage of information needed for the word2vec look up table.
 *
 * @author Adam Gibson
 */
public interface VocabCache  {





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
	 * Load vocab
	 */
	void loadVocab();
	
 
	/**
	 * Vocab exists already
	 * @return
	 */
	boolean vocabExists();
	
    /**
     * Saves the vocab: this allow for reuse of word frequencies	
     */
	void saveVocab();
	
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
     */
    void iterateSample(VocabWord w1,VocabWord w2,AtomicLong nextRandom);

    /**
     * Returns all of the words in the vocab
     * @returns all the words in the vocab
     */
    Collection<String> words();

    /**
     * Reset the weights of the cache
     */
    void resetWeights();

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
    long totalWordOccurrences();

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
     * Inserts the word as a vocab word
     * (it gets the vocab word from the internal token store).
     * Note that the index must be set on the token.
     * @param word the word to add to the vocab
     */
    void putVocabWord(String word);

    /**
     * Returns the number of words in the cache
     * @return the number of words in the cache
     */
    int numWords();


    /**
     * Count of documents a word appeared in
     * @param word the number of documents the word appeared in
     * @return
     */
    int docAppearedIn(String word);

    /**
     * Increment the document count
     * @param word the word to increment by
     * @param howMuch
     */
    void incrementDocCount(String word,int howMuch);


    /**
     * Set the count for the number of documents the word appears in
     * @param word the word to set the count for
     * @param count the count of the word
     */
    void setCountForDoc(String word,int count);

    /**
     * Returns the total of number of documents encountered in the corpus
     * @return the total number of docs in the corpus
     */
    int totalNumberOfDocs();


    /**
     * Increment the doc count
     */
    void incrementTotalDocCount();

    /**
     * Increment the doc count
     * @param  by the number to increment by
     */
    void incrementTotalDocCount(int by);

    /**
     * All of the tokens in the cache, (not necessarily apart of the vocab)
     * @return the tokens for this cache
     */
    Collection<VocabWord> tokens();


    /**
     * Adds a token
     * to the cache
     * @param word the word to add
     */
    void addToken(VocabWord word);

    /**
     * Returns the token (again not necessarily in the vocab)
     * for this word
     * @param word the word to get the token for
     * @return the vocab word for this token
     */
    VocabWord tokenFor(String word);

    /**
     * Returns whether the cache
     * contains this token or not
     * @param token the token to tes
     * @return whether the token exists in
     * the cache or not
     *
     */
    boolean hasToken(String token);


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
