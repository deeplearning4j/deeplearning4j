/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.models.word2vec;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;


import com.google.common.util.concurrent.AtomicDouble;
import lombok.Getter;
import lombok.NonNull;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.Word2VecConfiguration;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.word2vec.wordstore.VocabularyHolder;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.sentenceiterator.StreamLineIterator;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



/**
 * Leveraging a 3 layer neural net with a softmax approach as output,
 * converts a word based on its context and the training examples in to a
 * numeric vector
 * @author Adam Gibson
 *
 */
public class Word2Vec extends WordVectorsImpl {


    protected static final long serialVersionUID = -2367495638286018038L;


    @Getter protected transient Word2VecConfiguration configuration = new Word2VecConfiguration();
    protected transient TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
    protected transient SentenceIterator sentenceIter;
    protected transient DocumentIterator docIter;
    protected transient TextVectorizer vectorizer;
    protected transient InvertedIndex invertedIndex;

    protected transient VocabularyHolder vocabularyHolder;
    protected transient  RandomGenerator g;

    protected transient int workers = Runtime.getRuntime().availableProcessors();


    protected int batchSize = 1000;
    protected double sample = 0;
    protected long totalWords = 1;
    //learning rate
    protected AtomicDouble alpha = new AtomicDouble(0.025);

    //context to use for gathering word frequencies
    protected int window = 5;

    protected static final Logger log = LoggerFactory.getLogger(Word2Vec.class);
    protected boolean shouldReset = true;
    //number of iterations to run
    protected int numIterations = 1;
    public final static String UNK = "UNK";
    protected long seed = 123;
    protected boolean saveVocab = false;
    protected double minLearningRate = 0.01;
    protected int learningRateDecayWords = 10000;

    // used to store number of total words occurencies, so it can be passed from vocab building
    private transient AtomicLong totalProperWordsCount = new AtomicLong(0);

    // lines counter
    final AtomicLong totalLines = new AtomicLong(0);

    protected boolean useAdaGrad = false;


    public Word2Vec() {}

    public TextVectorizer getVectorizer() {
        return vectorizer;
    }

    public void setVectorizer(TextVectorizer vectorizer) {
        this.vectorizer = vectorizer;
    }


    /**
     * This method adds all unknown words to vocabulary. Known words get their counters updated.
     * And returns number of words being added/incremented in vocabulary
     *
     * @param tokens list of strings received from Tokenizer
     */
    protected int fillVocabulary(List<String> tokens) {
        AtomicInteger wordsAdded = new AtomicInteger(0);
        for (String token: tokens) {
            // check word against stopList

            if (stopWords !=null && stopWords.contains(token)) continue;

            if (!vocabularyHolder.containsWord(token)) {
                vocabularyHolder.addWord(token);
                wordsAdded.incrementAndGet();
            } else {
                vocabularyHolder.incrementWordCounter(token);
                wordsAdded.incrementAndGet();
            }
        }
        return wordsAdded.get();
    }

    /**
     * Returns sentence as list of word from vocabulary.
     *
     * @param tokens - list of tokens from sentence
     * @return
     */
    protected List<VocabWord> digitizeSentence(List<String> tokens) {
        List<VocabWord> result = new ArrayList<>(tokens.size());
        for (String token: tokens) {
            if (stopWords != null && stopWords.contains(token)) continue;

            VocabWord word = vocab.wordFor(token);
            if (word != null) result.add(word);
        }
        return result;
    }

    /**
     * This is plan-maxima feature. I want this model to be updatable after major training is done.
     * Will be added soon(tm), and there's nothing that could stop it, if most of the words in sentence is already in vocab.
     *
     * @param sentence
     */
    protected void fit(String sentence) {
        // TODO: implementation is in process

        // since this method can be executed on model restored from scratches, we have to be sure taht TokenizerFactory is set to something reasonable
        if (tokenizerFactory == null) throw new IllegalStateException("TokenizerFactory should be declared before fit() is called!");

        /*
            we should check if this model contains huffman tree info or word counters.
            as measure of this state we could use number of words occurences in vocab.
            For proper model it should be greater then number of words in vocab
          */

        Tokenizer tokenizer = tokenizerFactory.create(sentence);
    }


    /**
     * Train the model
     */
    public void fit() throws IOException {

        if (sentenceIter == null && docIter == null) throw new IllegalStateException("At least one iterator is needed for model fit()");


        // this queue is used for cross-thread communication between VectoCalculationsThreads and AsyncIterator thread
        final LinkedBlockingQueue<List<VocabWord>> sentences = new LinkedBlockingQueue<>();

        buildVocab();

        // totalWordsCount is used for learningRate decay at VectorCalculationsThreads
        final long totalWordsCount =this.totalProperWordsCount.get();

        /*
         vector representation part
        */

        // initialize vector table
        // resetWeights is absolutely required after vocab transfer, due to algo internals.
        log.info("Building matrices & resetting weights...");
        lookupTable.resetWeights();

        // at this moment sentence iterator should be reset and read once again
        // since there's no reason to save intermediate data. On huge corpus this will take like 50% of initial space, so why just not reset iterator, and read once again in cycle?

        int iteration = 1;

        final long maxLines = totalLines.get();
        // TODO: this should be done in cycle, corresponding to the number of iterations. Slow for large data, but that's proper way to do this.
        while (iteration <= numIterations) {
            log.info("Starting async iterator...");
            // resetting line counter, since we're going to roll over iterator once again
            totalLines.set(0);
            final AtomicLong wordsCounter = new AtomicLong(0);
            AsyncIteratorDigitizer roller = new AsyncIteratorDigitizer(sentenceIter, sentences, totalLines);
            roller.start();

            log.info("Starting vectorization process...");
            final VectorCalculationsThread[] threads = new VectorCalculationsThread[workers];
            // start processing threads
            for (int x = 0; x < workers; x++) {
                threads[x] = new VectorCalculationsThread(x, maxLines, iteration, wordsCounter, totalWordsCount, totalLines, sentences);
                threads[x].start();
            }

            try {
                // block untill all lines are read at AsyncIteratorDigitizer
                roller.join();
            } catch (Exception e) {
                e.printStackTrace();
            }

            // wait untill all vector calculation threads are finished
            for (int x = 0; x < workers; x++) {
                try {
                    threads[x].join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            log.info("Iteration: " + iteration + "; Lines vectorized so far: " + totalLines.get());
            iteration++;
        }

        log.info("Vectorization accomplished.");
    }



    protected void addWords(List<VocabWord> sentence,AtomicLong nextRandom,List<VocabWord> currMiniBatch) {
        for (VocabWord word : sentence) {
            if(word == null)
                continue;
            // The subsampling randomly discards frequent words while keeping the ranking same
            if (sample > 0) {
                double numDocs =  vectorizer.index().numDocuments();
                double ran = (Math.sqrt(word.getWordFrequency() / (sample * numDocs)) + 1)
                        * (sample * numDocs) / word.getWordFrequency();

                if (ran < (nextRandom.get() & 0xFFFF) / (double) 65536) {
                    continue;
                }

                currMiniBatch.add(word);
            }
            else
                currMiniBatch.add(word);
        }
    }


    /**
     * Build the binary tree
     * Reset the weights
     */
    public void setup() {

        log.info("Building binary tree");
        buildBinaryTree();
        log.info("Resetting weights");
        if(shouldReset)
            resetWeights();

    }


    /**
     * Builds the vocabulary for training
     */
    public boolean buildVocab() {
        if (sentenceIter == null && docIter == null) throw new IllegalStateException("At least one iterator is needed for model fit()");

        /*
            vocabulary building part of task
         */

        // SentenceIterator should be reset, so we'll be assured that previous iterator usage wont mess us
        sentenceIter.reset();

        log.info("Building vocabulary...");
        while (sentenceIter.hasNext()) {
            Tokenizer tokenizer = tokenizerFactory.create(sentenceIter.nextSentence());
            // insert new words in vocabulary, and update counters for already known words
            // as result it returns number of words being added or incremented in the vocab
            int wordsAdded =  this.fillVocabulary(tokenizer.getTokens());

            // at this moment we're pretty sure that each word from this sentence is already in vocabulary and all counters are updated
            if (wordsAdded > 0) totalLines.incrementAndGet();

            if (totalLines.get() % 100000 == 0) log.info("" + totalLines.get() + " lines parsed. Vocab size: " + vocabularyHolder.numWords());
        }

        log.info("" + totalLines.get() + " lines parsed. Vocab size: " + vocabularyHolder.numWords());
        vocabularyHolder.truncateVocabulary(minWordFrequency);


        log.info("Total truncated vocab size: " + vocabularyHolder.numWords());

        totalProperWordsCount.set(vocabularyHolder.totalWordsBeyondLimit() * numIterations);

        // as soon as vocab is built, we can switch back to VocabCache
        // please note: huffman tree building is hidden inside transfer method
        // please note: after this line VocabularyHolder is empty, since all data is moved into VocabCache
        vocabularyHolder.transferBackToVocabCache(vocab);


        return false;
    }



    /**
     * Train on a list of vocab words
     * @param sentence the list of vocab words to train on
     */
    public void trainSentence(final List<VocabWord> sentence,AtomicLong nextRandom,double alpha) {
        if(sentence == null || sentence.isEmpty())
            return;
        for(int i = 0; i < sentence.size(); i++) {
            nextRandom.set(nextRandom.get() * 25214903917L + 11);
            skipGram(i, sentence, (int) nextRandom.get() % window,nextRandom,alpha);
        }
    }


    /**
     * Train via skip gram
     * @param i
     * @param sentence
     */
    public void skipGram(int i,List<VocabWord> sentence, int b,AtomicLong nextRandom,double alpha) {

        final VocabWord word = sentence.get(i);
        if(word == null || sentence.isEmpty()) return;

        int end =  window * 2 + 1 - b;
        for(int a = b; a < end; a++) {
            if(a != window) {
                int c = i - window + a;
                if(c >= 0 && c < sentence.size()) {
                    VocabWord lastWord = sentence.get(c);
                    iterate(word,lastWord,nextRandom,alpha);
                }
            }
        }
    }

    /**
     * Train the word vector
     * on the given words
     * @param w1 the first word to fit
     */
    public void  iterate(VocabWord w1, VocabWord w2,AtomicLong nextRandom,double alpha) {
        lookupTable.iterateSample(w1,w2,nextRandom,alpha);
    }




    /* Builds the binary tree for the word relationships */
    protected void buildBinaryTree() {
        log.info("Constructing priority queue");
        Huffman huffman = new Huffman(vocab().vocabWords());
        huffman.build();

        log.info("Built tree");

    }

    /* reinit weights */
    protected void resetWeights() {
        lookupTable.resetWeights();
    }

    @SuppressWarnings("unchecked")
    protected void readStopWords() {
        if(this.stopWords != null)
            return;
        this.stopWords = StopWords.getStopWords();

    }

    /**
     * Note that calling a setter on this
     * means assumes that this is a training continuation
     * and therefore weights should not be reset.
     * @param sentenceIter
     */
    public void setSentenceIter(SentenceIterator sentenceIter) {
        this.sentenceIter = sentenceIter;
        this.shouldReset = false;
    }

    /**
     * restart training on next fit().
     * Use when sentence iterator is set for new training.
     */
    public void resetWeightsOnSetup() {
        this.shouldReset = true;
    }

    public int getWindow() {
        return window;
    }
    public List<String> getStopWords() {
        return stopWords;
    }
    public synchronized SentenceIterator getSentenceIter() {
        return sentenceIter;
    }
    public TokenizerFactory getTokenizerFactory() {
        return tokenizerFactory;
    }
    public void setTokenizerFactory(TokenizerFactory tokenizerFactory) {
        this.tokenizerFactory = tokenizerFactory;
    }

    public static class Builder {
        protected int minWordFrequency = 1;
        protected int layerSize = 50;
        protected SentenceIterator iter;
        protected List<String> stopWords = StopWords.getStopWords();
        protected int window = 5;
        protected TokenizerFactory tokenizerFactory;
        protected VocabCache vocabCache;
        protected DocumentIterator docIter;
        protected double lr = 2.5e-2;
        protected int iterations = 1;
        protected long seed = 123;
        protected boolean saveVocab = false;
        protected int batchSize = 1000;
        protected int learningRateDecayWords = 10000;
        protected boolean useAdaGrad = false;
        protected TextVectorizer textVectorizer;
        protected double minLearningRate = 1e-2;
        protected double negative = 0;
        protected double sampling = 1e-5;
        protected int workers = Runtime.getRuntime().availableProcessors();
        protected InvertedIndex index;
        protected WeightLookupTable lookupTable;
        protected boolean hugeModelExpected = false;
        private Word2VecConfiguration configuration = new Word2VecConfiguration();


        public Builder lookupTable(@NonNull WeightLookupTable lookupTable) {
            this.lookupTable = lookupTable;
            return this;
        }

        public Builder() {

        }

        /**
         * Whole configuration is transferred via Word2VecConfiguration bean
         *
         * @param conf
         */
        public Builder(@NonNull Word2VecConfiguration conf) {
            this.iterations = conf.getIterations();
            this.hugeModelExpected = conf.isHugeModelExpected();
            this.useAdaGrad = conf.isUseAdaGrad();
            this.minWordFrequency = conf.getMinWordFrequency();
            this.lr = conf.getLearningRate();
            this.learningRateDecayWords = conf.getLearningRateDecayWords();
            this.negative = conf.getNegative();
            this.sampling = conf.getSampling();
            this.minLearningRate = conf.getMinLearningRate();
            this.window = conf.getWindow();
            this.seed = conf.getSeed();

            this.configuration = conf;
        }

        /**
         * This method is deprecated, since InvertedIndex isn't used for vocab building anymore. We're rolling over iterator instead.
         *
         * @param index
         * @return
         */
        @Deprecated
        public Builder index(InvertedIndex index) {
            this.index = index;
            return this;
        }

        public Builder workers(int workers) {
            this.workers = workers;
            return this;
        }

        public Builder sampling(double sample) {
            this.sampling = sample;
            return this;
        }


        public Builder negativeSample(double negative) {
            this.negative = negative;
            return this;
        }

        public Builder minLearningRate(double minLearningRate) {
            this.minLearningRate = minLearningRate;
            return this;
        }

        public Builder useAdaGrad(boolean useAdaGrad) {
            this.useAdaGrad = useAdaGrad;
            return this;
        }

        /**
         * This method is deprecated, since vectorizer isn't used for vocab building anymore
         * @param textVectorizer
         * @return
         */
        @Deprecated
        public Builder vectorizer(TextVectorizer textVectorizer) {
            this.textVectorizer = textVectorizer;
            return this;
        }

        public Builder learningRateDecayWords(int learningRateDecayWords) {
            this.learningRateDecayWords = learningRateDecayWords;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder saveVocab(boolean saveVocab){
            this.saveVocab = saveVocab;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder iterations(int iterations) {
            this.iterations = iterations;
            return this;
        }


        public Builder learningRate(double lr) {
            this.lr = lr;
            return this;
        }


        public Builder iterate(@NonNull DocumentIterator iter) {
            this.docIter = iter;
            return this;
        }

        /**
         * All words in this VocabCache will be treated as SPECIAL words, and they won't be affected by minWordFrequency argument.
         * That's good way to inject words from human-marked corpora, so you'll be sure they won't be missed.
         *
         * @param cache
         * @return
         */
        public Builder vocabCache(@NonNull VocabCache cache) {
            this.vocabCache = cache;
            return this;
        }

        public Builder minWordFrequency(int minWordFrequency) {
            this.minWordFrequency = minWordFrequency;
            return this;
        }

        public Builder tokenizerFactory(@NonNull TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }



        public Builder layerSize(int layerSize) {
            this.layerSize = layerSize;
            return this;
        }

        public Builder stopWords(List<String> stopWords) {
            this.stopWords = stopWords;
            return this;
        }

        public Builder windowSize(int window) {
            this.window = window;
            return this;
        }

        public Builder iterate(@NonNull SentenceIterator iter) {
            this.iter = iter;
            return this;
        }

        /**
         * If you're going for huge model built from scratches, you can use this option to avoid excessive memory use during vocab building.
         * Setting this option to true will force vocab to be trashed perodically, based on each word occurence dynamic
         *
         * @param reallyExpected
         * @return
         */
        public Builder hugeModelExpected(boolean reallyExpected) {
            this.hugeModelExpected = reallyExpected;
            return this;
        }




        public Word2Vec build() {
                /*
                    if there's DocumentIterator instead of SentenceIterator provided, flatten it down to StreamLineIterator and use it as SentenceIterator at fit()
                    since anyway we're working on single sentence level, without other options.
                  */
                if (iter == null && docIter != null) {
                    this.iter = new StreamLineIterator.Builder(docIter)
                            .setFetchSize(100)
                            .build();
                }

                Word2Vec ret = new Word2Vec();
                ret.alpha.set(lr);
                ret.sentenceIter = iter;
                ret.window = window;
                ret.useAdaGrad = useAdaGrad;
                ret.minLearningRate = minLearningRate;
                ret.vectorizer = textVectorizer;
                ret.stopWords = stopWords;
                ret.minWordFrequency = minWordFrequency;
                ret.setVocab(vocabCache);
                ret.docIter = docIter;
                ret.minWordFrequency = minWordFrequency;
                ret.numIterations = iterations;
                ret.seed = seed;
                ret.numIterations = iterations;
                ret.saveVocab = saveVocab;
                ret.batchSize = batchSize;
                ret.sample = sampling;
                ret.workers = workers;
                ret.invertedIndex = index;
                ret.lookupTable = lookupTable;

                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }

                if(vocabCache == null) {
                    vocabCache = new InMemoryLookupCache();

                    ret.setVocab(vocabCache);
                }

                if(lookupTable == null) {
                    lookupTable = new InMemoryLookupTable.Builder().negative(negative)
                            .useAdaGrad(useAdaGrad).lr(lr).cache(vocabCache)
                            .vectorLength(layerSize).build();
                }
                ret.lookupTable = lookupTable;
                ret.tokenizerFactory = tokenizerFactory;

                // VocabularyHolder is used ONLY for fit() purposes, as intermediate data storage
                if (this.vocabCache!= null)
                    // if VocabCache is set, build VocabHolder on top of it. Just for compatibility
                    // please note: all words in VocabCache will be treated as SPECIAL, so they wont be affected by minWordFrequency
                    ret.vocabularyHolder = new VocabularyHolder.Builder()
                            .externalCache(vocabCache)
                            .hugeModelExpected(hugeModelExpected)
                            .minWordFrequency(minWordFrequency)
                            .scavengerActivationThreshold(this.configuration.getScavengerActivationThreshold())
                            .scavengerRetentionDelay(this.configuration.getScavengerRetentionDelay())
                            .build();
                else ret.vocabularyHolder = new VocabularyHolder.Builder()
                        .hugeModelExpected(hugeModelExpected)
                        .minWordFrequency(minWordFrequency)
                        .scavengerActivationThreshold(this.configuration.getScavengerActivationThreshold())
                        .scavengerRetentionDelay(this.configuration.getScavengerRetentionDelay())
                        .build();



            this.configuration.setLearningRate(lr);
            this.configuration.setLayersSize(layerSize);
            this.configuration.setHugeModelExpected(hugeModelExpected);
            this.configuration.setWindow(window);
            this.configuration.setMinWordFrequency(minWordFrequency);
            this.configuration.setIterations(iterations);
            this.configuration.setSeed(seed);
            this.configuration.setBatchSize(batchSize);
            this.configuration.setLearningRateDecayWords(learningRateDecayWords);
            this.configuration.setMinLearningRate(minLearningRate);
            this.configuration.setSampling(this.sampling);
            this.configuration.setUseAdaGrad(useAdaGrad);
            this.configuration.setNegative(negative);

            ret.configuration = this.configuration;

            return ret;
        }
    }

    /**
     * This class is used to fetch data from iterator in background thread, and convert it to List<VocabularyWord>
     *
     * It becomes very usefull if text processing pipeline behind iterator is complex, and we're not loading data from simple text file with whitespaces as separator.
     * Since this method allows you to hide preprocessing latency in background.
     */
    private class AsyncIteratorDigitizer extends Thread implements Runnable {
        private final SentenceIterator iterator;
        private final LinkedBlockingQueue<List<VocabWord>> buffer;
        private final AtomicLong linesCounter;
        private final int limitUpper = 10000;
        private final int limitLower = 5000;

        public AsyncIteratorDigitizer(SentenceIterator iterator, LinkedBlockingQueue<List<VocabWord>> buffer, AtomicLong linesCounter) {
            this.iterator = iterator;
            this.buffer = buffer;
            this.linesCounter = linesCounter;
            this.setName("AsyncIteratorReader thread");

            this.iterator.reset();
        }

        @Override
        public void run() {
            while (this.iterator.hasNext()) {

                // if buffered level is below limitLower, we're going to fetch limitUpper number of strings from fetcher
                if (buffer.size() < limitLower) {
                    AtomicInteger linesLoaded = new AtomicInteger(0);
                    while (linesLoaded.getAndIncrement() < limitUpper && this.iterator.hasNext()) {
                        Tokenizer tokenizer = Word2Vec.this.tokenizerFactory.create(this.iterator.nextSentence());

                        // convert text sentence to list of word IDs from vocab
                        List<VocabWord> list = Word2Vec.this.digitizeSentence(tokenizer.getTokens());
                        if (list != null && !list.isEmpty()) {
                            buffer.add(list);
                        }
                        linesLoaded.incrementAndGet();
                    }
                } else {
                    try {
                        Thread.sleep(50);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        }
    }

    /**
     * VectorCalculationsThreads are used for vector calculations, and work together with AsyncIteratorDigitizer.
     * Basically, all they do is just transfer of digitized sentences into math layer.
     *
     * Please note, they do not iterate the sentences over and over, each sentence processed only once.
     * Training corpus iteration is implemented in fit() method.
     *
     */
    private class VectorCalculationsThread extends Thread implements Runnable {
        private final int threadId;
        private final long linesLimit;
        private final int iterationId;
        private final AtomicLong wordsCounter;
        private final long totalWordsCount;
        private final AtomicLong totalLines;
        private final LinkedBlockingQueue<List<VocabWord>> sentences;

        public VectorCalculationsThread(int threadId, long linesLimit, int iteration, AtomicLong wordsCounter, long totalWordsCount, AtomicLong linesCounter, LinkedBlockingQueue<List<VocabWord>> buffer) {
            this.threadId = threadId;
            this.linesLimit = linesLimit;
            this.iterationId = iteration;
            this.wordsCounter = wordsCounter;
            this.totalWordsCount = totalWordsCount;
            this.totalLines = linesCounter;
            this.sentences = buffer;
            this.setName("VectorCalculationsThread " + this.threadId);
        }

        @Override
        public void run() {
            final AtomicLong nextRandom = new AtomicLong(5);
            while (totalLines.get() < this.linesLimit  || sentences.size() > 0) {
                try {
                    // get current sentence as list of VocabularyWords
                    List<VocabWord> sentence = sentences.poll(2L, TimeUnit.SECONDS);

                    // TODO: investigate, if fix needed here to become iteration-dependent, not line-position
                    double alpha = Math.max(minLearningRate, Word2Vec.this.alpha.get() * (1 - (1.0 * wordsCounter.get() / (double) totalWordsCount)));

                    if (sentence != null && !sentence.isEmpty()) {

                        Word2Vec.this.trainSentence(sentence, nextRandom, alpha);

                        // increment processed word count, please note: this affects learningRate decay
                        wordsCounter.addAndGet(sentence.size());
                    } //else log.warn("sentence is null");

                    // increment processed lines count
                    totalLines.incrementAndGet();
                    if (totalLines.get() % 100000 == 0) log.info("Iteration: " + this.iterationId+ "; Words vectorized so far: " + wordsCounter.get() + ";  Lines vectorized so far: " + totalLines.get() + "; learningRate: " + alpha);
                } catch (Exception  e) {
                    e.printStackTrace();
                    throw new RuntimeException(e);
                }
            }
        }
    }
}
