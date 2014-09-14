package org.deeplearning4j.models.word2vec.wordstore.inmemory;

import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.util.Index;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * In memory lookup cache for smaller datasets
 *
 * @author Adam Gibson
 */
public class InMemoryLookupCache implements VocabCache,Serializable {

    private Index wordIndex = new Index();
    private Counter<String> wordFrequencies = new Counter<>();
    private Map<String,VocabWord> vocabs = new HashMap<>();
    private Map<String,INDArray> vectors = new HashMap<>();
    private Map<Integer,INDArray> codes = new HashMap<>();
    private int codeLength = 0;
    private int vectorLength = 50;

    public InMemoryLookupCache(int vectorLength) {
        this.vectorLength = vectorLength;
    }

    /**
     * Increment the count for the given word
     *
     * @param word the word to increment the count for
     */
    @Override
    public void incrementWordCount(String word) {
        incrementWordCount(word,1);
    }

    /**
     * Increment the count for the given word by
     * the amount increment
     *
     * @param word      the word to increment the count for
     * @param increment the amount to increment by
     */
    @Override
    public void incrementWordCount(String word, int increment) {
        wordFrequencies.incrementCount(word,1);

    }

    /**
     * Returns the number of times the word has occurred
     *
     * @param word the word to retrieve the occurrence frequency for
     * @return 0 if hasn't occurred or the number of times
     * the word occurs
     */
    @Override
    public int wordFrequency(String word) {
        return (int) wordFrequencies.getCount(word);
    }

    /**
     * Returns true if the cache contains the given word
     *
     * @param word the word to check for
     * @return
     */
    @Override
    public boolean containsWord(String word) {
        return wordFrequencies.containsKey(word);
    }

    /**
     * Returns the word contained at the given index or null
     *
     * @param index the index of the word to get
     * @return the word at the given index
     */
    @Override
    public String wordAtIndex(int index) {
        return (String) wordIndex.get(index);
    }

    /**
     * Returns the index of a given word
     *
     * @param word the index of a given word
     * @return the index of a given word or -1
     * if not found
     */
    @Override
    public int indexOf(String word) {
        return wordIndex.indexOf(word);
    }

    /**
     * @param codeIndex
     * @param code
     */
    @Override
    public void putCode(int codeIndex, INDArray code) {
        codeLength = code.length();
        codes.put(codeIndex,code);
    }

    /**
     * Loads the co-occurrences for the given codes
     *
     * @param codes the codes to load
     * @return an ndarray of code.length by layerSize
     */
    @Override
    public INDArray loadCodes(int[] codes) {
        INDArray load = Nd4j.create(codes.length,codeLength);
        for(int i = 0; i < load.rows(); i++) {
            load.putRow(i,this.codes.get(codes[i]));
        }
        return load;
    }

    /**
     * Returns all of the vocab word nodes
     *
     * @return
     */
    @Override
    public Collection<VocabWord> vocabWords() {
        return vocabs.values();
    }

    /**
     * The total number of word occurrences
     *
     * @return the total number of word occurrences
     */
    @Override
    public int totalWordOccurrences() {
        return (int) wordFrequencies.totalCount();
    }

    /**
     * Inserts a word vector
     *
     * @param word   the word to insert
     * @param vector the vector to insert
     */
    @Override
    public void putVector(String word, INDArray vector) {
        vectors.put(word,vector);
    }

    /**
     * @param word
     * @return
     */
    @Override
    public INDArray vector(String word) {
        return vectors.get(word);
    }

    /**
     * @param word
     * @return
     */
    @Override
    public VocabWord wordFor(String word) {
        return vocabs.get(word);
    }

    /**
     * @param index
     * @param word
     */
    @Override
    public void addWordToIndex(int index, String word) {
        if(!wordFrequencies.containsKey(word))
            wordFrequencies.incrementCount(word,1);
        if(!vocabs.containsKey(word))
            vocabs.put(word,new VocabWord(1,vectorLength));
        wordIndex.add(word);
    }

    /**
     * @param word
     * @param vocabWord
     */
    @Override
    public void putVocabWord(String word, VocabWord vocabWord) {
        vocabs.put(word,vocabWord);
    }

    /**
     * Returns the number of words in the cache
     *
     * @return the number of words in the cahce
     */
    @Override
    public int numWords() {
        return vocabs.size();
    }
}
