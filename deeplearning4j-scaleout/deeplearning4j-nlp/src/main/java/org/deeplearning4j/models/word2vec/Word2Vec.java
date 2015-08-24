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
import java.util.concurrent.atomic.AtomicLong;


import akka.actor.ActorSystem;
import com.google.common.base.Function;
import com.google.common.util.concurrent.AtomicDouble;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.parallel.Parallelization;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.invertedindex.LuceneInvertedIndex;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.stopwords.StopWords;
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

    protected transient TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
    protected transient SentenceIterator sentenceIter;
    protected transient DocumentIterator docIter;
    protected int batchSize = 1000;
    protected double sample = 0;
    protected long totalWords = 1;
    //learning rate
    protected AtomicDouble alpha = new AtomicDouble(0.025);

    //context to use for gathering word frequencies
    protected int window = 5;
    protected transient  RandomGenerator g;
    protected static final Logger log = LoggerFactory.getLogger(Word2Vec.class);
    protected boolean shouldReset = true;
    //number of iterations to run
    protected int numIterations = 1;
    public final static String UNK = "UNK";
    protected long seed = 123;
    protected boolean saveVocab = false;
    protected double minLearningRate = 0.01;
    protected transient TextVectorizer vectorizer;
    protected int learningRateDecayWords = 10000;
    protected InvertedIndex invertedIndex;
    protected boolean useAdaGrad = false;
    protected int workers = Runtime.getRuntime().availableProcessors();

    public Word2Vec() {}

    public TextVectorizer getVectorizer() {
        return vectorizer;
    }

    public void setVectorizer(TextVectorizer vectorizer) {
        this.vectorizer = vectorizer;
    }

    /**
     * Train the model
     */
    public void fit() throws IOException {
        boolean loaded = buildVocab();
        //save vocab after building
        if (!loaded && saveVocab)
            vocab().saveVocab();
        if (stopWords == null)
            readStopWords();


        log.info("Training word2vec multithreaded");

        if (sentenceIter != null)
            sentenceIter.reset();
        if (docIter != null)
            docIter.reset();


        int[] docs = vectorizer.index().allDocs();

        if(docs.length < 1) {
            vectorizer.fit();
        }

        docs = vectorizer.index().allDocs();
        if(docs.length < 1) {
            throw new IllegalStateException("No documents found");
        }


        totalWords = vectorizer.numWordsEncountered();
        if(totalWords < 1)
            throw new IllegalStateException("Unable to train, total words less than 1");

        totalWords *= numIterations;



        log.info("Processing sentences...");


        AtomicLong numWordsSoFar = new AtomicLong(0);
        final AtomicLong nextRandom = new AtomicLong(5);
        ExecutorService exec = new ThreadPoolExecutor(Runtime.getRuntime().availableProcessors(),
                Runtime.getRuntime().availableProcessors(),
                0L, TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<Runnable>(), new RejectedExecutionHandler() {
            @Override
            public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                executor.submit(r);
            }
        });

        final Queue<List<VocabWord>> batch2 = new ConcurrentLinkedDeque<>();
        vectorizer.index().eachDoc(new Function<List<VocabWord>, Void>() {
            @Override
            public Void apply(List<VocabWord> input) {
                List<VocabWord> batch = new ArrayList<>();
                addWords(input, nextRandom, batch);
                if(!batch.isEmpty()) {
                  batch2.add(batch);
                }

                return null;
            }
        },exec);

        exec.shutdown();
        try {
            exec.awaitTermination(1,TimeUnit.DAYS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }




        ActorSystem actorSystem = ActorSystem.create();

        for(int i = 0; i < numIterations; i++)
            doIteration(batch2,numWordsSoFar,nextRandom,actorSystem);
        actorSystem.shutdown();


    }



    private void doIteration(Collection<List<VocabWord>> batch2,final AtomicLong numWordsSoFar,final AtomicLong nextRandom,ActorSystem actorSystem) {
        final AtomicLong lastReported = new AtomicLong(System.currentTimeMillis());
        Parallelization.iterateInParallel(batch2, new Parallelization.RunnableWithParams<List<VocabWord>>() {
            @Override
            public void run(List<VocabWord> sentence, Object[] args) {
                double alpha = Math.max(minLearningRate, Word2Vec.this.alpha.get() *
                        (1 - (1.0 * numWordsSoFar.get() / (double) totalWords)));
                long now = System.currentTimeMillis();
                long diff = Math.abs(now - lastReported.get());
                if (numWordsSoFar.get() > 0 && diff > 1000) {
                    lastReported.set(now);
                    log.info("Words so far " + numWordsSoFar.get() + " with alpha at " + alpha);
                }


                trainSentence(sentence, nextRandom, alpha);
                numWordsSoFar.set(numWordsSoFar.get() + sentence.size());


            }
        },actorSystem);
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
        readStopWords();

        if(vocab().vocabExists()) {
            log.info("Loading vocab...");
            vocab().loadVocab();
            lookupTable.resetWeights();
            return true;
        }


        if(invertedIndex == null)
            invertedIndex = new LuceneInvertedIndex.Builder()
                    .cache(vocab()).stopWords(stopWords)
                    .build();
        //vectorizer will handle setting up vocab meta data
        if(vectorizer == null) {
            vectorizer = new TfidfVectorizer.Builder().index(invertedIndex)
                    .cache(vocab()).iterate(docIter).iterate(sentenceIter).batchSize(batchSize)
                    .minWords(minWordFrequency).stopWords(stopWords)
                    .tokenize(tokenizerFactory).build();

            vectorizer.fit();

        }

        //includes unk
        else if(vocab().numWords() < 2)
            vectorizer.fit();

        setup();

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
        if(word == null || sentence.isEmpty())
            return;

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

        public Builder lookupTable(WeightLookupTable lookupTable) {
            this.lookupTable = lookupTable;
            return this;
        }

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


        public Builder iterate(DocumentIterator iter) {
            this.docIter = iter;
            return this;
        }

        public Builder vocabCache(VocabCache cache) {
            this.vocabCache = cache;
            return this;
        }

        public Builder minWordFrequency(int minWordFrequency) {
            this.minWordFrequency = minWordFrequency;
            return this;
        }

        public Builder tokenizerFactory(TokenizerFactory tokenizerFactory) {
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

        public Builder iterate(SentenceIterator iter) {
            this.iter = iter;
            return this;
        }




        public Word2Vec build() {

            if(iter == null) {
                Word2Vec ret = new Word2Vec();
                ret.window = window;
                ret.alpha.set(lr);
                ret.vectorizer = textVectorizer;
                ret.stopWords = stopWords;
                ret.setVocab(vocabCache);
                ret.numIterations = iterations;
                ret.minWordFrequency = minWordFrequency;
                ret.seed = seed;
                ret.saveVocab = saveVocab;
                ret.batchSize = batchSize;
                ret.useAdaGrad = useAdaGrad;
                ret.minLearningRate = minLearningRate;
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


                ret.docIter = docIter;
                ret.lookupTable = lookupTable;
                ret.tokenizerFactory = tokenizerFactory;

                return ret;
            }

            else {
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
                return ret;
            }

        }
    }

}
