package org.deeplearning4j.models.word2vec.wordstore.inmemory;

import com.google.common.util.concurrent.AtomicDouble;
import it.unimi.dsi.util.XorShift64StarRandomGenerator;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.plot.Tsne;
import org.deeplearning4j.plot.dropwizard.RenderApplication;
import org.deeplearning4j.text.movingwindow.Util;
import org.deeplearning4j.util.Index;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * In memory lookup cache for smaller datasets
 *
 * @author Adam Gibson
 */
public class InMemoryLookupCache implements VocabCache,Serializable {

    private Index wordIndex = new Index();
    private boolean useAdaGrad = false;
    private Counter<String> wordFrequencies = Util.parallelCounter();
    private Counter<String> docFrequencies = Util.parallelCounter();
    private Map<String,VocabWord> vocabs = new ConcurrentHashMap<>();
    private Map<String,VocabWord> tokens = new ConcurrentHashMap<>();
    private Map<Integer,INDArray> codes = new ConcurrentHashMap<>();
    private INDArray syn0,syn1;
    private int vectorLength = 50;
    private transient RandomGenerator rng = new XorShift64StarRandomGenerator(123);
    private AtomicLong totalWordOccurrences = new AtomicLong(0);
    private AtomicDouble lr = new AtomicDouble(1e-1);
    double[] expTable = new double[1000];
    static double MAX_EXP = 6;
    private long seed = 123;
    private int numDocs = 0;
    //negative sampling table
    private INDArray table,syn1Neg;
    private double negative = 0;


    public InMemoryLookupCache(int vectorLength,boolean useAdaGrad,double lr,RandomGenerator gen,double negative) {
        this.vectorLength = vectorLength;
        this.useAdaGrad = useAdaGrad;
        this.lr.set(lr);
        this.rng = gen;
        addToken(new VocabWord(1.0,Word2Vec.UNK));
        addWordToIndex(0, Word2Vec.UNK);
        putVocabWord(Word2Vec.UNK);
        this.negative = negative;
        initExpTable();



    }




    private void initNegative() {
        if(negative > 0) {
            syn1Neg = Nd4j.zeros(syn0.shape());
            makeTable(10000,0.75);
        }
    }


    private void initExpTable() {
        for (int i = 0; i < expTable.length; i++) {
            double tmp =   FastMath.exp((i / (double) expTable.length * 2 - 1) * MAX_EXP);
            expTable[i]  = tmp / (tmp + 1.0);
        }
    }




    /**
     * Iterate on the given 2 vocab words
     *
     * @param w1 the first word to iterate on
     * @param w2 the second word to iterate on
     * @param nextRandom next random for sampling
     */
    @Override
    public  void iterateSample(VocabWord w1, VocabWord w2,AtomicLong nextRandom,double alpha) {
        if(w2 == null || w2.getIndex() < 0)
            return;
        //current word vector
        INDArray l1 = this.syn0.slice(w2.getIndex());


        //error for current word and context
        INDArray neu1e = Nd4j.create(vectorLength);


        double avgChange = 0.0f;




        for(int i = 0; i < w1.getCodeLength(); i++) {
            int code = w1.getCodes().get(i);
            int point = w1.getPoints().get(i);
            if(point >= syn0.rows() || point < 0)
                throw new IllegalStateException("Illegal point " + point);
            //other word vector

            INDArray syn1 = this.syn1.slice(point);


            double dot = Nd4j.getBlasWrapper().dot(l1,syn1);

            if(dot < -MAX_EXP || dot >= MAX_EXP)
                continue;


            int idx = (int) ((dot + MAX_EXP) * ((double) expTable.length / MAX_EXP / 2.0));
            if(idx >= expTable.length)
                continue;

            //score
            double f =  expTable[idx];
            //gradient
            double g = (1 - code - f) * (useAdaGrad ?  w1.getLearningRate(i,alpha) : alpha);




            avgChange += g;
            if(syn0.data().dataType() == DataBuffer.DOUBLE) {
                Nd4j.getBlasWrapper().axpy(g, syn1, neu1e);
                Nd4j.getBlasWrapper().axpy(g, l1, syn1);
            }
            else {
                Nd4j.getBlasWrapper().axpy((float) g, syn1, neu1e);
                Nd4j.getBlasWrapper().axpy((float) g, l1, syn1);
            }


        }


        int target = w1.getIndex();
        int label;
        //negative sampling
        if(negative > 0)
            for (int d = 0; d < negative + 1; d++) {
                if (d == 0) {

                    label = 1;
                } else {
                    nextRandom.set(nextRandom.get() * 25214903917L + 11);
                    target = table.getInt((int) (nextRandom.get() >> 16) % table.length());
                    if (target == 0)
                        target = (int) nextRandom.get() % (numWords() - 1) + 1;
                    if (target == w1.getIndex())
                        continue;
                    label = 0;
                }

                double f = Nd4j.getBlasWrapper().dot(l1,syn1Neg.slice(target));
                double g;
                if (f > MAX_EXP)
                    g = (label - 1) * (useAdaGrad ?  w1.getLearningRate(target,alpha) : alpha);
                else if (f < -MAX_EXP)
                    g = (label - 0) * (useAdaGrad ?  w1.getLearningRate(target,alpha) : alpha);
                else
                    g = (label - expTable[(int)((f + MAX_EXP) * (expTable.length / MAX_EXP / 2))]) *  (useAdaGrad ?  w1.getLearningRate(target,alpha) : alpha);
                if(syn0.data().dataType() == DataBuffer.DOUBLE)
                    Nd4j.getBlasWrapper().axpy(g,neu1e,l1);
                else
                    Nd4j.getBlasWrapper().axpy((float) g,neu1e,l1);

                if(syn0.data().dataType() == DataBuffer.DOUBLE)
                    Nd4j.getBlasWrapper().axpy(g,syn1Neg,l1);
                else
                    Nd4j.getBlasWrapper().axpy((float) g,syn1Neg,l1);
            }


        avgChange /=  w1.getCodes().size();


        if(useAdaGrad) {
            if(syn0.data().dataType() == DataBuffer.DOUBLE)
                Nd4j.getBlasWrapper().axpy(avgChange,neu1e,l1);
            else
                Nd4j.getBlasWrapper().axpy((float) avgChange,neu1e,l1);


        }
        else {
            if(syn0.data().dataType() == DataBuffer.DOUBLE)
                Nd4j.getBlasWrapper().axpy(1.0,neu1e,l1);

            else
                Nd4j.getBlasWrapper().axpy(1.0f,neu1e,l1);

        }




    }

    /**
     * Iterate on the given 2 vocab words
     *
     * @param w1 the first word to iterate on
     * @param w2 the second word to iterate on
     */
    @Override
    public  void iterate(VocabWord w1, VocabWord w2) {
        if(w2.getIndex() < 0)
            return;
        //current word vector
        INDArray l1 = this.syn0.slice(w2.getIndex());

        //error for current word and context
        INDArray neu1e = Nd4j.create(vectorLength);


        double avgChange = 0.0f;


        double alpha = this.lr.get();

        for(int i = 0; i < w1.getCodeLength(); i++) {
            int code = w1.getCodes().get(i);
            int point = w1.getPoints().get(i);
            if(point >= syn0.rows() || point < 0)
                throw new IllegalStateException("Illegal point " + point);
            //other word vector
            INDArray syn1 = this.syn1.slice(point);


            double dot = Nd4j.getBlasWrapper().dot(l1,syn1);

            if(dot < -MAX_EXP || dot >= MAX_EXP)
                continue;


            int idx = (int) ((dot + MAX_EXP) * ((double) expTable.length / MAX_EXP / 2.0));
            if(idx >= expTable.length)
                continue;

            //score
            double f =  expTable[idx];
            //gradient
            double g = (1 - code - f) * (useAdaGrad ?  w1.getLearningRate(i,alpha) : alpha);

            avgChange += g;
            if(syn0.data().dataType() == DataBuffer.DOUBLE) {
                Nd4j.getBlasWrapper().axpy(g, syn1, neu1e);
                Nd4j.getBlasWrapper().axpy(g, l1, syn1);
            }
            else {
                Nd4j.getBlasWrapper().axpy((float) g, syn1, neu1e);
                Nd4j.getBlasWrapper().axpy((float) g, l1, syn1);
            }
        }




        avgChange /=  w1.getCodes().size();


        if(useAdaGrad) {
            if(syn0.data().dataType() == DataBuffer.DOUBLE)
                Nd4j.getBlasWrapper().axpy(avgChange,neu1e,l1);
            else
                Nd4j.getBlasWrapper().axpy((float) avgChange,neu1e,l1);


        }
        else {
            if(syn0.data().dataType() == DataBuffer.DOUBLE)
                Nd4j.getBlasWrapper().axpy(1.0,neu1e,l1);

            else
                Nd4j.getBlasWrapper().axpy(1.0f,neu1e,l1);

        }




    }




    private void makeTable(int tableSize,double power) {
        int vocabSize = numWords();
        table = Nd4j.create(new FloatBuffer(tableSize));
        double trainWordsPow = 0.0;
        for(String word : words()) {
            trainWordsPow += Math.pow(wordFrequency(word), power);
        }

        int wordIdx = 0;
        double d1 = Math.pow(wordFrequency(wordAtIndex(wordIdx)),power) / trainWordsPow;
        for(int i = 0; i < tableSize; i++) {
            table.putScalar(i,wordIdx);
            double mul = i * 1.0 / (double) tableSize;
            if(mul > d1) {
                wordIdx++;
                String wordAtIndex = wordAtIndex(wordIdx);
                if(wordAtIndex == null)
                    continue;
                d1 += Math.pow(wordFrequency(wordAtIndex),power) / trainWordsPow;

            }

            if(wordIdx >= vocabSize)
                wordIdx = vocabSize - 1;

        }

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

        syn0  = Nd4j.rand(new int[]{vocabs.size(),vectorLength},rng).subi(0.5).divi(vectorLength);
        putVector(Word2Vec.UNK,Nd4j.rand(1,vectorLength,rng).subi(0.5).divi(vectorLength));

        syn1 = Nd4j.create(syn0.shape());
        initNegative();

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
     * Loads the co-occurrences for the given codes
     *
     * @param codes the codes to load
     * @return an ndarray of code.length by layerSize
     */
    @Override
    public INDArray loadCodes(int[] codes) {
        return syn1.getRows(codes);
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
     * Inserts a word vector
     *
     * @param word   the word to insert
     * @param vector the vector to insert
     */
    @Override
    public void putVector(String word, INDArray vector) {
        if(word == null)
            throw new IllegalArgumentException("No null words allowed");
        if(vector == null)
            throw new IllegalArgumentException("No null vectors allowed");
        int idx = indexOf(word);
        syn0.slice(idx).assign(vector);

    }

    /**
     * @param word
     * @return
     */
    @Override
    public INDArray vector(String word) {
        if(word == null)
            return null;
        int idx = indexOf(word);
        if(idx < 0)
            idx = indexOf(Word2Vec.UNK);
        return syn0.getRow(idx);
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

    @Override
    public Iterator<INDArray> vectors() {
        return new WeightIterator();
    }


    private  class WeightIterator implements Iterator<INDArray> {
        private int currIndex = 0;

        @Override
        public boolean hasNext() {
            return currIndex < syn0.rows();
        }

        @Override
        public INDArray next() {
            INDArray ret = syn0.slice(currIndex);
            currIndex++;
            return ret;
        }

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
    public void iterateSentence() {

    }

    @Override
    public void plotVocab(Tsne tsne) {
        try {
            List<String> plot = new ArrayList<>();
            for(String s : words()) {
                plot.add(s);
            }
            tsne.plot(syn0, 2, plot);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        try {
            RenderApplication.main(null);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Render the words via tsne
     */
    @Override
    public void plotVocab() {
        Tsne tsne = new Tsne.Builder()
                .normalize(false).setFinalMomentum(0.8f)
                .setMaxIter(1000).build();
        try {
            List<String> plot = new ArrayList<>();
            for(String s : words()) {
                plot.add(s);
            }
            tsne.plot(syn0,2,plot);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void loadVocab() {
        InMemoryLookupCache cache = SerializationUtils.readObject(new File("ser"));
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

    public INDArray getSyn0() {
        return syn0;
    }

    public void setSyn0(INDArray syn0) {
        this.syn0 = syn0;
    }

    public INDArray getSyn1() {
        return syn1;
    }

    public void setSyn1(INDArray syn1) {
        this.syn1 = syn1;
    }


    public static class Builder {
        private int vectorLength = 100;
        private boolean useAdaGrad = false;
        private double lr = 0.025;
        private RandomGenerator gen = new XorShift64StarRandomGenerator(123);
        private long seed = 123;
        private double negative = 0;





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



        public InMemoryLookupCache build() {
            InMemoryLookupCache ret =  new InMemoryLookupCache(vectorLength,useAdaGrad,lr,gen,negative);

            return ret;
        }
    }


}
