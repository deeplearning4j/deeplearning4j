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

package org.deeplearning4j.models.glove;

import akka.actor.ActorSystem;
import com.google.common.collect.Lists;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.LegacyTfidfVectorizer;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.parallel.Parallelization;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.invertedindex.LuceneInvertedIndex;
import org.deeplearning4j.text.movingwindow.Util;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Glove by socher et. al
 *
 * @author Adam Gibson
 */
@Deprecated
public class LegacyGlove  extends WordVectorsImpl<VocabWord> {

    private transient SentenceIterator sentenceIterator;
    private transient TextVectorizer textVectorizer;
    private transient TokenizerFactory tokenizerFactory;
    private double learningRate = 0.05;
    private double xMax = 0.75;
    private int windowSize = 15;
    private CoOccurrences coOccurrences;
    private boolean stem = false;
    protected Queue<Pair<Integer,List<Pair<VocabWord,VocabWord>>>> jobQueue = new LinkedBlockingDeque<>();
    private int batchSize = 1000;
    private int minWordFrequency = 5;
    private double maxCount = 100;
    public final static String UNK = Word2Vec.DEFAULT_UNK;
    private int iterations = 5;
    private static final Logger log = LoggerFactory.getLogger(Glove.class);
    private boolean symmetric = true;
    private transient Random gen;
    private boolean shuffle = true;
    private transient Random shuffleRandom;
    private int numWorkers = Runtime.getRuntime().availableProcessors();

    private LegacyGlove(){}

    public LegacyGlove(VocabCache cache, SentenceIterator sentenceIterator, TextVectorizer textVectorizer, TokenizerFactory tokenizerFactory, GloveWeightLookupTable lookupTable, int layerSize, double learningRate, double xMax, int windowSize, CoOccurrences coOccurrences, List<String> stopWords, boolean stem,int batchSize,int minWordFrequency,double maxCount,int iterations,boolean symmetric,Random gen,boolean shuffle,long seed,int numWorkers) {
        this.numWorkers = numWorkers;
        this.gen = gen;
        this.vocab = cache;
        this.layerSize = layerSize;
        this.shuffle = shuffle;
        this.sentenceIterator = sentenceIterator;
        this.textVectorizer = textVectorizer;
        this.tokenizerFactory = tokenizerFactory;
        this.lookupTable = lookupTable;
        this.learningRate = learningRate;
        this.xMax = xMax;
        this.windowSize = windowSize;
        this.coOccurrences = coOccurrences;
        this.stopWords = stopWords;
        this.stem = stem;
        this.batchSize = batchSize;
        this.minWordFrequency = minWordFrequency;
        this.maxCount = maxCount;
        this.iterations = iterations;
        this.symmetric = symmetric;
        shuffleRandom = Nd4j.getRandom();
    }

    public void fit() {
        boolean cacheFresh = false;

        if(vocab() == null) {
            cacheFresh  = true;
            setVocab(new InMemoryLookupCache());
        }

        if(textVectorizer == null && cacheFresh) {
            InvertedIndex index = new LuceneInvertedIndex(vocab(),false,"glove-index");
            textVectorizer = new LegacyTfidfVectorizer.Builder().tokenize(tokenizerFactory).index(index)
                    .cache(vocab()).iterate(sentenceIterator).minWords(minWordFrequency)
                    .stopWords(stopWords).stem(stem).build();

            textVectorizer.fit();
        }

        if(sentenceIterator != null)
            sentenceIterator.reset();

        if(coOccurrences == null) {
            coOccurrences = new CoOccurrences.Builder()
                    .cache(vocab()).iterate(sentenceIterator).symmetric(symmetric)
                    .tokenizer(tokenizerFactory).windowSize(windowSize)
                    .build();

            coOccurrences.fit();

        }

        if(lookupTable == null) {
            lookupTable = new GloveWeightLookupTable.Builder()
                    .cache(textVectorizer.vocab()).lr(learningRate)
                    .vectorLength(layerSize).maxCount(maxCount)
                   .build();
        }


        if(lookupTable().getSyn0() == null)
            lookupTable().resetWeights();
        final List<Pair<String,String>> pairList = coOccurrences.coOccurrenceList();
        if(shuffle)
            Collections.shuffle(pairList,new java.util.Random());



        final AtomicInteger countUp = new AtomicInteger(0);
        final Counter<Integer> errorPerIteration = Util.parallelCounter();
        log.info("Processing # of co occurrences " + coOccurrences.numCoOccurrences());
        for(int i = 0; i < iterations; i++) {
            final AtomicInteger processed = new AtomicInteger(coOccurrences.numCoOccurrences());
            doIteration(i, pairList, errorPerIteration, processed, countUp);
            log.info("Processed " + countUp.doubleValue() + " out of " + (pairList.size() * iterations) + " error was " + errorPerIteration.getCount(i));

        }


    }


    public void doIteration(final int i,List<Pair<String,String>> pairList, final Counter<Integer> errorPerIteration,final AtomicInteger processed,final AtomicInteger countUp) {
        log.info("Iteration " + i);
        if(shuffle)
            Collections.shuffle(pairList,new java.util.Random());
        List<List<Pair<String,String>>> miniBatches = Lists.partition(pairList,batchSize);
        ActorSystem actor = ActorSystem.create();
        Parallelization.iterateInParallel(miniBatches,new Parallelization.RunnableWithParams<List<Pair<String, String>>>() {
            @Override
            public void run(List<Pair<String, String>> currentItem, Object[] args) {
                List<Pair<VocabWord,VocabWord>> send = new ArrayList<>();
                for (Pair<String, String> next : currentItem) {
                    String w1 = next.getFirst();
                    String w2 = next.getSecond();
                    VocabWord vocabWord = vocab().wordFor(w1);
                    VocabWord vocabWord1 = vocab().wordFor(w2);
                    send.add(new Pair<>(vocabWord, vocabWord1));

                }

                jobQueue.add(new Pair<>(i, send));
            }
        },actor);



        actor.shutdown();

        Parallelization.runInParallel(numWorkers,new Runnable() {
            @Override
            public void run() {
                while(processed.get() > 0 || !jobQueue.isEmpty()) {
                    Pair<Integer,List<Pair<VocabWord,VocabWord>>> work = jobQueue.poll();
                    if(work == null)
                        continue;
                    List<Pair<VocabWord,VocabWord>> batch = work.getSecond();

                    for(Pair<VocabWord,VocabWord> pair : batch) {
                        VocabWord w1 = pair.getFirst();
                        VocabWord w2 = pair.getSecond();
                        double weight = getCount(w1.getWord(),w2.getWord());
                        if(weight <= 0) {
                            countUp.incrementAndGet();
                            processed.decrementAndGet();
                            continue;

                        }
                        //errorPerIteration.incrementCount(work.getFirst(),lookupTable().iterateSample(w1, w2, weight));
                        countUp.incrementAndGet();
                        if(countUp.get() % 10000 == 0)
                            log.info("Processed " + countUp.get() + " co occurrences");
                        processed.decrementAndGet();
                    }




                }
            }
        },true);
    }


    /**
     * Load a glove model from an input stream.
     * The format is:
     * word num1 num2....
     * @param is the input stream to read from for the weights
     * @param biases the bias input stream
     * @return the loaded model
     * @throws IOException if one occurs
     */
    public static LegacyGlove load(InputStream is,InputStream biases) throws IOException {
        LineIterator iter = IOUtils.lineIterator(is,"UTF-8");
        LegacyGlove glove = new LegacyGlove();
        Map<String,float[]> wordVectors = new HashMap<>();
        int count = 0;
        while(iter.hasNext()) {
            String line = iter.nextLine().trim();
            if(line.isEmpty())
                continue;
            String[] split = line.split(" ");
            String word = split[0];
            if(glove.vocab() == null)
                glove.setVocab(new InMemoryLookupCache());

            if(glove.lookupTable() == null) {
                glove.lookupTable = new GloveWeightLookupTable.Builder()
                        .cache(glove.vocab()).vectorLength(split.length - 1)
                        .build();

            }

            if(word.isEmpty())
                continue;
            float[] read = read(split,glove.lookupTable().layerSize());
            if(read.length < 1)
                continue;

            VocabWord w1 = new VocabWord(1,word);
            w1.setIndex(count);
            glove.vocab().addToken(w1);
            glove.vocab().addWordToIndex(count, word);
            glove.vocab().putVocabWord(word);
            wordVectors.put(word,read);
            count++;



        }

        glove.lookupTable().setSyn0(weights(glove, wordVectors));



        iter.close();

        glove.lookupTable().setBias(Nd4j.read(biases));

        return glove;

    }




    private static INDArray weights(LegacyGlove glove,Map<String,float[]> data) {
        INDArray ret = Nd4j.create(data.size(),glove.lookupTable().layerSize());
        for(String key : data.keySet()) {
            INDArray row = Nd4j.create(Nd4j.createBuffer(data.get(key)));
            if(row.length() != glove.lookupTable().layerSize())
                continue;
            if(glove.vocab().indexOf(key) >= data.size())
                continue;
            ret.putRow(glove.vocab().indexOf(key), row);
        }
        return ret;
    }


    private static float[] read(String[] split,int length) {
        float[] ret = new float[length];
        for(int i = 1; i < split.length; i++) {
            ret[i - 1] = Float.parseFloat(split[i]);
        }
        return ret;
    }


    public double getCount(String w1,String w2) {
        return coOccurrences.getCoOCurreneCounts().getCount(w1,w2);
    }

    public CoOccurrences getCoOccurrences() {
        return coOccurrences;
    }

    public void setCoOccurrences(CoOccurrences coOccurrences) {
        this.coOccurrences = coOccurrences;
    }







    @Override
    public GloveWeightLookupTable lookupTable() {
        return (GloveWeightLookupTable) lookupTable;
    }

    public void setLookupTable(GloveWeightLookupTable lookupTable) {
        this.lookupTable = lookupTable;
    }

    public static class Builder<T> {
        private VocabCache vocabCache;
        private SentenceIterator sentenceIterator;
        private TextVectorizer textVectorizer;
        private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        private GloveWeightLookupTable weightLookupTable;
        private int layerSize = 300;
        private double learningRate = 0.05;
        private double xMax = 0.75;
        private int windowSize = 5;
        private CoOccurrences coOccurrences;
        private List<String> stopWords = StopWords.getStopWords();
        private boolean stem = false;
        private int batchSize = 100;
        private int minWordFrequency = 5;
        private double maxCount = 100;
        private int iterations = 5;
        private boolean symmetric = true;
        private boolean shuffle = true;
        private long seed = 123;
        private int numWorkers = Runtime.getRuntime().availableProcessors();
        private org.nd4j.linalg.api.rng.Random gen = Nd4j.getRandom();


        public Builder numWorkers(int numWorkers) {
            this.numWorkers = numWorkers;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder shuffle(boolean shuffle) {
            this.shuffle = shuffle;
            return this;
        }
        public Builder rng(org.nd4j.linalg.api.rng.Random gen) {
            this.gen = gen;
            return this;
        }

        public Builder symmetric(boolean symmetric) {
            this.symmetric = symmetric;
            return this;
        }

        public Builder iterations(int iterations) {
            this.iterations = iterations;
            return this;
        }

        public Builder maxCount(double maxCount) {
            this.maxCount = maxCount;
            return this;
        }

        public Builder minWordFrequency(int minWordFrequency) {
            this.minWordFrequency = minWordFrequency;
            return this;
        }

        public Builder cache(VocabCache vocabCache) {
            this.vocabCache = vocabCache;
            return this;
        }

        public Builder iterate(SentenceIterator sentenceIterator) {
            this.sentenceIterator = sentenceIterator;
            return this;
        }

        public Builder vectorizer(TextVectorizer textVectorizer) {
            this.textVectorizer = textVectorizer;
            return this;
        }

        public Builder tokenizer(TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        public Builder weights(GloveWeightLookupTable weightLookupTable) {
            this.weightLookupTable = weightLookupTable;
            return this;
        }

        public Builder layerSize(int layerSize) {
            this.layerSize = layerSize;
            return this;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder xMax(double xMax) {
            this.xMax = xMax;
            return this;
        }

        public Builder windowSize(int windowSize) {
            this.windowSize = windowSize;
            return this;
        }

        public Builder coOccurrences(CoOccurrences coOccurrences) {
            this.coOccurrences = coOccurrences;
            return this;
        }

        public Builder stopWords(List<String> stopWords) {
            this.stopWords = stopWords;
            return this;
        }

        public Builder stem(boolean stem) {
            this.stem = stem;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public LegacyGlove build() {
            return new LegacyGlove(vocabCache, sentenceIterator, textVectorizer, tokenizerFactory, weightLookupTable, layerSize, learningRate, xMax, windowSize, coOccurrences, stopWords, stem, batchSize,minWordFrequency,maxCount,iterations,symmetric,gen,shuffle,seed,numWorkers);
        }
    }

}
