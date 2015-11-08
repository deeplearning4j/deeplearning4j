package org.deeplearning4j.models.word2vec;

import com.google.common.util.concurrent.AtomicDouble;
import lombok.Getter;
import lombok.NonNull;
import org.apache.commons.lang.math.RandomUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.VocabularyHolder;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Classic w2v implementation, suitable for handling large amounts of data
 * Internally its using own vocab mechanics, as soon as computation is complete - all data is transferred into provided InMemoryLookupCache
 *
 * Based on original google w2v & dl4j w2v implementation by Adam Gibson
 * Backed by nd4j math, by Adam Gibson
 * @author raver119@gmail.com
 */
public class StandaloneWord2Vec extends Word2Vec {
    // vector for unknown words
   // public static final String UNK = "UNK";

    @Getter
    private VocabularyHolder vocabularyHolder;
    private int minWordFrequency;
    @Getter  private int numThreads;


    private LinkedBlockingQueue<List<VocabWord>> sentences = new LinkedBlockingQueue<>();

    private AtomicInteger totalWordsProcessed = new AtomicInteger(0);
    private AtomicLong totalLines = new AtomicLong(0);


    private Logger logger = LoggerFactory.getLogger(StandaloneWord2Vec.class);

    /**
     * Hidden dummy constructor. Word2Vec.Builder() should be used to create W2V instance.
     */
    private StandaloneWord2Vec() {

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

            VocabWord word = vocab.wordFor(token); //vocabularyHolder.getVocabularyWordByString(token);
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
    private void trainSentence(String sentence) {
        // TODO: to be implemented later
    }


    /**
     * Build vector-space representation for provided data.
     */
    @Override
    public void fit() {
        /*
            vocabulary part, move it to another scope, to avoid GC leaks
         */
        logger.info("Building vocabulary...");
        while (sentenceIter.hasNext()) {
            Tokenizer tokenizer = tokenizerFactory.create(sentenceIter.nextSentence());
            // insert new words in vocabulary, and update counters for already known words
            // as result it returns number of words being added or incremented in the vocab
            int wordsAdded =  this.fillVocabulary(tokenizer.getTokens());

            // at this moment we're pretty sure that each word from this sentence is already in vocabulary and all counters are updated
            if (wordsAdded > 0) totalLines.incrementAndGet();

            if (totalLines.get() % 100000 == 0) logger.info("" + totalLines.get() + " lines parsed. Vocab size: " + vocabularyHolder.numWords());
        }

        logger.info("" + totalLines.get() + " lines parsed. Vocab size: " + vocabularyHolder.numWords());
        vocabularyHolder.truncateVocabulary(minWordFrequency);

        // totalWordsCount is used for learningRate decay at VectorCalculationsThreads
        final long totalWordsCount = vocabularyHolder.totalWordsBeyondLimit() * numIterations;

        logger.info("Total truncated vocab size: " + vocabularyHolder.numWords());
        // as soon as vocab is built, we can switch back to VocabCache
        // please note: huffman tree building is hidden inside transfer method
        // please note: after this line VocabularyHolder is empty, since all data is moved into VocabCache
        vocabularyHolder.transferBackToVocabCache(vocab);

        /*
         vector representation part
        */

        // initialize vector table
        // resetWeights is absolutely required after vocab transfer, due to algo internals.
        logger.info("Building matrices & resetting weights...");
        lookupTable.resetWeights();

        // at this moment sentence iterator should be reset and read once again
        // since there's no reason to save intermediate data. On huge corpus this will take like 50% of initial space, so why just not reset iterator, and read once again in cycle?

        int iteration = 1;

        final long maxLines = totalLines.get();
        // TODO: this should be done in cycle, corresponding to the number of iterations. Slow for large data, but that's proper way to do this.
        while (iteration <= numIterations) {
            logger.info("Starting async iterator...");
            // resetting line counter, since we're going to roll over iterator once again
            totalLines.set(0);
            final AtomicLong wordsCounter = new AtomicLong(0);
            AsyncIteratorDigitizer roller = new AsyncIteratorDigitizer(sentenceIter, sentences, totalLines);
            roller.start();

            logger.info("Starting vectorization process...");
            final VectorCalculationsThread[] threads = new VectorCalculationsThread[numThreads];
            // start processing threads
            for (int x = 0; x < numThreads; x++) {
                threads[x] = new VectorCalculationsThread(x, maxLines, iteration, wordsCounter, totalWordsCount);
                threads[x].start();
            }

            try {
                // block untill all lines are read at AsyncIteratorDigitizer
                roller.join();
            } catch (Exception e) {
                e.printStackTrace();
            }

            // wait untill all vector calculation threads are finished
            for (int x = 0; x < numThreads; x++) {
                try {
                    threads[x].join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            logger.info("Iteration: " + iteration + "; Lines vectorized so far: " + totalLines.get());
            iteration++;
        }

        logger.info("Vectorization accomplished.");
    }



    public static class Builder {
        private long seed;
        private SentenceIterator iterator;
        private TokenizerFactory tokenizerFactory;
        private VocabularyHolder vocabCache;
        private int minWordFrequency;
        private int dimensions;
        private List<String> stopList;
        private int iterations;
        private double learningRate = 0.025d;
        private int windowSize;
        private int numThreads = Runtime.getRuntime().availableProcessors();
        private VocabCache externalCache;
        private WeightLookupTable lookupTable;


        public Builder() {

        }

        /**
         * Sets the seed for Randum Numbers Generator.
         *
         * @param seed
         * @return
         */
        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        /**
         * Sets the SentenceIterator that contains training corpus.
         *
         * @param iterator
         * @return
         */
        public Builder iterator(@NonNull SentenceIterator iterator) {
            this.iterator = iterator;
            return this;
        }

        /**
         * Sets the TokenizerFactory, used for sentence tokenization.
         *
         * @param factory
         * @return
         */
        public Builder tokenizerFactory(@NonNull TokenizerFactory factory) {
            this.tokenizerFactory = factory;
            return this;
        }


        /**
         * Sets the minimum word frequency threshold. All words with frequency below this number in training corpus will be discarded.
         *
         * @param minWordFrequency
         * @return
         */
        public Builder minWordFrequency(int minWordFrequency) {
            this.minWordFrequency = minWordFrequency;
            return this;
        }

        /**
         * Sets WeightLookupTable, used for calculations
         *
         * @param table
         * @return
         */
        public Builder lookupTable(WeightLookupTable table) {
            this.lookupTable = table;
            return this;
        }


        /**
         * Sets the number of dimensions in output vectors.
         *
         * @param dimensions
         * @return
         */
        public Builder layerSize(int dimensions) {
            this.dimensions = dimensions;
            return this;
        }

        /**
         * Provides list of stop words. Optional.
         *
         * @param stopList
         * @return
         */
        public Builder stopList(@NonNull List<String> stopList) {
            this.stopList = stopList;
            return this;
        }

        /**
         * Sets number of iterations over training corpus
         * @param iterations
         * @return
         */
        public Builder iterations(int iterations) {
            this.iterations = iterations;
            return this;
        }

        /**
         * Sets training window size. Important argument.
         *
         * @param windowsSize
         * @return
         */
        public Builder windowSize(int windowsSize) {
            this.windowSize = windowsSize;
            return this;
        }

        public Builder numThreads(int numThreads) {
            this.numThreads = numThreads;
            return this;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder vocabCache(@NonNull VocabCache cache) {
            this.externalCache = cache;
            return this;
        }

        /**
         * Used for test purposes only, keep it protected please
         *
         * @param holder
         * @return
         */
        protected Builder vocabHolder(@NonNull VocabularyHolder holder) {
            this.vocabCache = holder;
            return this;
        }

        public StandaloneWord2Vec build() {
            if (this.seed == 0L) this.seed = RandomUtils.nextLong();
//            if (this.tokenizerFactory == null) throw new IllegalStateException("TokenizerFactory wasnt set.");
            if (this.stopList == null) this.stopList = new ArrayList<>();
            if (this.iterations < 1) this.iterations = 1;
            if (this.windowSize < 1) this.windowSize= 5;
            if (this.dimensions < 50) this.dimensions = 50;

            if (this.externalCache == null) this.externalCache = new InMemoryLookupCache();

            // VocabularyHolder is used ONLY for fit() purposes, as intermediate data storage
            if (this.vocabCache == null) {
                if (this.externalCache != null)
                    // if VocabCache is set, build VocabHolder on top of it. Just for compatibility
                    this.vocabCache = new VocabularyHolder(this.externalCache);
                else this.vocabCache = new VocabularyHolder();
            }


            StandaloneWord2Vec vec = new StandaloneWord2Vec();
            vec.seed = this.seed;
            vec.sentenceIter = this.iterator;
            vec.tokenizerFactory = this.tokenizerFactory;
            vec.vocabularyHolder = this.vocabCache;
            vec.minWordFrequency = this.minWordFrequency;
            vec.layerSize = this.dimensions;
            vec.stopWords = this.stopList;
            vec.numIterations = this.iterations;
            vec.window = this.windowSize;
            vec.numThreads = this.numThreads;
            vec.alpha.set(this.learningRate);
            vec.vocab = this.externalCache;
            vec.lookupTable = this.lookupTable;

            return vec;
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
                        Tokenizer tokenizer = StandaloneWord2Vec.this.tokenizerFactory.create(this.iterator.nextSentence());

                        // convert text sentence to list of word IDs from vocab
                        List<VocabWord> list = StandaloneWord2Vec.this.digitizeSentence(tokenizer.getTokens());
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

        public VectorCalculationsThread(int threadId, long linesLimit, int iteration, AtomicLong wordsCounter, long totalWordsCount) {
            this.threadId = threadId;
            this.linesLimit = linesLimit;
            this.iterationId = iteration;
            this.wordsCounter = wordsCounter;
            this.totalWordsCount = totalWordsCount;
            this.setName("VectorCalculationsThread " + threadId);
        }

        @Override
        public void run() {
            final AtomicLong nextRandom = new AtomicLong(5);
            while (totalLines.get() < this.linesLimit  || sentences.size() > 0) {
                try {
                    // get current sentence as list of VocabularyWords
                    List<VocabWord> sentence = sentences.poll(1L, TimeUnit.SECONDS);

                    // TODO: investigate, if fix needed here to become iteration-dependent, not line-position
                    double alpha = Math.max(minLearningRate, StandaloneWord2Vec.this.alpha.get() * (1 - (1.0 * wordsCounter.get() / (double) totalWordsCount)));

                    if (sentence != null && !sentence.isEmpty()) {
                        for(int i = 0; i < sentence.size(); i++) {
                            nextRandom.set(nextRandom.get() * 25214903917L + 11);
                            StandaloneWord2Vec.this.skipGram(i, sentence, (int) nextRandom.get() % window ,nextRandom,alpha);
                        }
                        // increment processed word count, please note: this affects learningRate decay
                        wordsCounter.addAndGet(sentence.size());
                    }

                     // increment processed lines count
                    totalLines.incrementAndGet();
                    if (totalLines.get() % 10000 == 0) logger.info("Iteration: " + this.iterationId+ "; Words vectorized so far: " + wordsCounter.get() + ";  Lines vectorized so far: " + totalLines.get() + "; learningRate: " + alpha);
                } catch (Exception  e) {
                    e.printStackTrace();
                    throw new RuntimeException(e);
                }
            }
        }
    }
}
