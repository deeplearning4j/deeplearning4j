package org.deeplearning4j.models.word2vec.wordstore;

import com.google.common.util.concurrent.AtomicDouble;
import it.unimi.dsi.util.XorShift64StarRandomGenerator;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.movingwindow.Util;
import org.deeplearning4j.util.Index;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * In memory lookup cache for smaller datasets
 *
 * @author Adam Gibson
 */
public abstract class BaseLookupCache implements VocabCache,Serializable {

    protected Index wordIndex = new Index();
    protected boolean useAdaGrad = false;
    protected Counter<String> wordFrequencies = Util.parallelCounter();
    protected Counter<String> docFrequencies = Util.parallelCounter();
    protected Map<String,VocabWord> vocabs = new ConcurrentHashMap<>();
    protected Map<String,VocabWord> tokens = new ConcurrentHashMap<>();
    protected Map<Integer,INDArray> codes = new ConcurrentHashMap<>();
    protected int vectorLength = 50;
    protected transient RandomGenerator rng = new XorShift64StarRandomGenerator(123);
    protected AtomicInteger totalWordOccurrences = new AtomicInteger(0);
    protected AtomicDouble lr = new AtomicDouble(1e-1);
    protected long seed = 123;
    protected int numDocs = 0;
    //negative sampling table
    protected double negative = 0;


    public BaseLookupCache(int vectorLength, boolean useAdaGrad, double lr, RandomGenerator gen, double negative) {
        this.vectorLength = vectorLength;
        this.useAdaGrad = useAdaGrad;
        this.lr.set(lr);
        this.rng = gen;
        addToken(new VocabWord(1.0,Word2Vec.UNK));
        addWordToIndex(0, Word2Vec.UNK);
        putVocabWord(Word2Vec.UNK);
        this.negative = negative;



    }



    /**
     * Returns all of the words in the vocab
     *
     * @returns all the words in the vocab
     */
    @Override
    public  synchronized Collection<String> words() {
        return vocabs.keySet();
    }

    /**
     * Reset the weights of the cache
     */
    @Override
    public void resetWeights() {
        this.rng = new MersenneTwister(seed);

    }

    /**
     * Increment the count for the given word
     *
     * @param word the word to increment the count for
     */
    @Override
    public  void incrementWordCount(String word) {
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
    public   void incrementWordCount(String word, int increment) {
        wordFrequencies.incrementCount(word,1);

        VocabWord token;
        if(hasToken(word))
            token = tokenFor(word);
        else
            token = new VocabWord(increment,word);
        //token and word in vocab will be same reference
        token.increment(increment);
        totalWordOccurrences.set(totalWordOccurrences.get() + increment);



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
        return vocabs.containsKey(word);
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
        codes.put(codeIndex,code);
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
    public long totalWordOccurrences() {
        return  totalWordOccurrences.get();
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
    public synchronized void addWordToIndex(int index, String word) {
        if(!wordFrequencies.containsKey(word))
            wordFrequencies.incrementCount(word,1);
        wordIndex.add(word,index);

    }

    /**
     * @param word
     */
    @Override
    public synchronized void putVocabWord(String word) {
        VocabWord token = tokenFor(word);
        addWordToIndex(token.getIndex(),word);
        if(!hasToken(word))
            throw new IllegalStateException("Unable to add token " + word + " when not already a token");
        vocabs.put(word,token);
        wordIndex.add(word,token.getIndex());

    }

    /**
     * Returns the number of words in the cache
     *
     * @return the number of words in the cache
     */
    @Override
    public synchronized int numWords() {
        return vocabs.size();
    }

    @Override
    public int docAppearedIn(String word) {
        return (int) docFrequencies.getCount(word);
    }

    @Override
    public void incrementDocCount(String word, int howMuch) {
        docFrequencies.incrementCount(word, howMuch);
    }

    @Override
    public void setCountForDoc(String word, int count) {
        docFrequencies.setCount(word, count);
    }

    @Override
    public int totalNumberOfDocs() {
        return numDocs;
    }

    @Override
    public void incrementTotalDocCount() {
        numDocs++;
    }

    @Override
    public void incrementTotalDocCount(int by) {
        numDocs += by;
    }

    @Override
    public Collection<VocabWord> tokens() {
        return tokens.values();
    }

    @Override
    public void addToken(VocabWord word) {
        tokens.put(word.getWord(),word);
    }

    @Override
    public VocabWord tokenFor(String word) {
        return tokens.get(word);
    }

    @Override
    public boolean hasToken(String token) {
        return tokenFor(token) != null;
    }

    @Override
    public void setLearningRate(double lr) {
        this.lr.set(lr);
    }


    protected abstract class WeightIterator implements Iterator<INDArray> {
        protected int currIndex = 0;

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }

    @Override
    public void saveVocab() {
        SerializationUtils.saveObject(this, new File("ser"));
    }

    @Override
    public boolean vocabExists() {
        return new File("ser").exists();
    }



    @Override
    public void loadVocab() {
        BaseLookupCache cache = SerializationUtils.readObject(new File("ser"));
        this.codes = cache.codes;
        this.vocabs = cache.vocabs;
        this.vectorLength = cache.vectorLength;
        this.wordFrequencies = cache.wordFrequencies;
        this.wordIndex = cache.wordIndex;
        this.tokens = cache.tokens;


    }





    public RandomGenerator getRng() {
        return rng;
    }

    public void setRng(RandomGenerator rng) {
        this.rng = rng;
    }




    public static abstract class Builder {
        protected int vectorLength = 100;
        protected boolean useAdaGrad = false;
        protected double lr = 0.025;
        protected RandomGenerator gen = new XorShift64StarRandomGenerator(123);
        protected long seed = 123;
        protected double negative = 0;





        public Builder negative(double negative) {
            this.negative = negative;
            return this;
        }

        public Builder vectorLength(int vectorLength) {
            this.vectorLength = vectorLength;
            return this;
        }

        public Builder useAdaGrad(boolean useAdaGrad) {
            this.useAdaGrad = useAdaGrad;
            return this;
        }


        public Builder lr(double lr) {
            this.lr = lr;
            return this;
        }

        public Builder gen(RandomGenerator gen) {
            this.gen = gen;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }



        public abstract BaseLookupCache build();
    }


}
