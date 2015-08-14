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

package org.deeplearning4j.spark.text;

import org.apache.spark.Accumulator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.stopwords.StopWords;

import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicLong;

import static org.deeplearning4j.spark.models.embeddings.word2vec.Word2VecVariables.*;
/**
 * A spark based text pipeline
 * with minimum word frequency and stop words
 *
 * @author Adam Gibson
 */
@SuppressWarnings("unchecked")
public class TextPipeline {
    //params
    private JavaRDD<String> corpusRDD;
    private int numWords;
    private int nGrams;
    private String tokenizer;
    private String tokenizerPreprocessor;
    private List<String> stopWords = new ArrayList<>();
    //Setup
    private JavaSparkContext sc;
    private Accumulator<Counter<String>> wordFreqAcc;
    private Broadcast<List<String>> stopWordBroadCast;
    // Return values
    private JavaRDD<Pair<List<String>, AtomicLong>> sentenceWordsCountRDD;
    private VocabCache vocabCache = new InMemoryLookupCache();
    private Broadcast<VocabCache> vocabCacheBroadcast;
    private JavaRDD<List<VocabWord>> vocabWordListRDD;
    private JavaRDD<AtomicLong> sentenceCountRDD;
    private long totalWordCount;

    // Getters
    public Accumulator<Counter<String>> getWordFreqAcc() {
        if (wordFreqAcc != null) {
            return wordFreqAcc;
        } else {
            throw new IllegalStateException("IllegalStateException: wordFreqAcc not set at TextPipline.");
        }
    }

    public Broadcast<VocabCache> getBroadCastVocabCache() throws IllegalStateException {
        if (vocabCache.numWords() > 0) {
            return vocabCacheBroadcast;
        } else {
            throw new IllegalStateException("IllegalStateException: VocabCache not set at TextPipline.");
        }
    }

    public VocabCache getVocabCache() throws IllegalStateException {
        if (vocabCache.numWords() > 0) {
            return vocabCache;
        } else {
            throw new IllegalStateException("IllegalStateException: VocabCache not set at TextPipline.");
        }
    }

    public JavaRDD<Pair<List<String>, AtomicLong>> getSentenceWordsCountRDD() {
        if (sentenceWordsCountRDD != null) {
            return sentenceWordsCountRDD;
        } else {
            throw new IllegalStateException("IllegalStateException: sentenceWordsCountRDD not set at TextPipline.");
        }
    }

    public JavaRDD<List<VocabWord>> getvocabWordListRDD() throws IllegalStateException {
        if (vocabWordListRDD != null) {
            return vocabWordListRDD;
        } else {
            throw new IllegalStateException("IllegalStateException: vocabWordListRDD not set at TextPipline.");
        }
    }

    public JavaRDD<AtomicLong> getSentenceCountRDD() throws IllegalStateException {
        if (sentenceCountRDD != null) {
            return sentenceCountRDD;
        } else {
            throw new IllegalStateException("IllegalStateException: sentenceCountRDD not set at TextPipline.");
        }
    }

    public Long getTotalWordCount() {
        if (totalWordCount != 0L) {
            return totalWordCount;
        } else {
            throw new IllegalStateException("IllegalStateException: totalWordCount not set at TextPipline.");
        }
    }

    // Constructor
    public TextPipeline(JavaRDD<String> corpusRDD, int numWords, int nGrams, String tokenizer,
                        String tokenPreprocessor, boolean removeStopWords) throws Exception {
        this.corpusRDD = corpusRDD;
        this.numWords = numWords;
        // TokenizerFunction Settings
        this.nGrams = nGrams;
        this.tokenizer = tokenizer;
        this.tokenizerPreprocessor = tokenPreprocessor;
        // Remove Stop words
        if (removeStopWords) {
            stopWords = StopWords.getStopWords();
        }
        // Setup all Spark variables
        setup();
    }

    // Constructor (For testing purposes)
    public TextPipeline(JavaRDD<String> corpusRDD) throws Exception {
        final JavaSparkContext sc = new JavaSparkContext(corpusRDD.context());
        final SparkConf conf = sc.getConf();
        int numWords = assignVar(NUM_WORDS, conf, Integer.class);
        int nGrams = assignVar(N_GRAMS, conf, Integer.class);
        String tokenizer = assignVar(TOKENIZER, conf, String.class);
        String tokenPreprocessor = assignVar(TOKEN_PREPROCESSOR, conf, String.class);
        boolean removeStop = assignVar(REMOVE_STOPWORDS, conf, Boolean.class);
        this.corpusRDD = corpusRDD;
        this.numWords = numWords;
        // TokenizerFunction Settings
        this.nGrams = nGrams;
        this.tokenizer = tokenizer;
        this.tokenizerPreprocessor = tokenPreprocessor;
        // Remove Stop words
        if (removeStop) {
            stopWords = StopWords.getStopWords();
        }
        // Setup all Spark variables
        setup();
    }

    private void setup() {
        // Set up accumulators and broadcast stopwords
        this.sc = new JavaSparkContext(corpusRDD.context());
        this.wordFreqAcc = sc.accumulator(new Counter<String>(), new WordFreqAccumulator());
        this.stopWordBroadCast = sc.broadcast(stopWords);
    }

    public JavaRDD<List<String>> tokenize() {
        if (corpusRDD == null) {
            throw new IllegalStateException("corpusRDD not assigned. Define TextPipeline with corpusRDD assigned.");
        }
        return corpusRDD.map(new TokenizerFunction(tokenizer, tokenizerPreprocessor, nGrams));
    }

    public JavaRDD<Pair<List<String>, AtomicLong>> updateAndReturnAccumulatorVal(JavaRDD<List<String>> tokenizedRDD) {
        // Update the 2 accumulators
        UpdateAccumulatorFunction accumulatorClassFunction = new UpdateAccumulatorFunction(stopWordBroadCast, wordFreqAcc);
        JavaRDD<Pair<List<String>, AtomicLong>> sentenceWordsCountRDD = tokenizedRDD.map(accumulatorClassFunction);

        // Loop through each element to update accumulator. Count does the same job (verified).
        sentenceWordsCountRDD.count();

        return sentenceWordsCountRDD;
    }

    private String filterMinWord(String stringToken, double tokenCount) {
        return (tokenCount < numWords) ? org.deeplearning4j.models.word2vec.Word2Vec.UNK : stringToken;
    }

    private void addTokenToVocabCache(String stringToken, Double tokenCount) {
        // Making string token into actual token if not already an actual token (vocabWord)
        VocabWord actualToken;
        if (vocabCache.hasToken(stringToken)) {
            actualToken = vocabCache.tokenFor(stringToken);
            actualToken.increment(tokenCount.intValue());
        } else {
            actualToken = new VocabWord(tokenCount, stringToken);
        }

        // Set the index of the actual token (vocabWord)
        // Put vocabWord into vocabs in InMemoryVocabCache
        boolean vocabContainsWord = vocabCache.containsWord(stringToken);
        if (!vocabContainsWord) {
            vocabCache.addToken(actualToken);
            int idx = vocabCache.numWords();
            actualToken.setIndex(idx);
            vocabCache.putVocabWord(stringToken);
        }
    }

    public void filterMinWordAddVocab(Counter<String> wordFreq) {

        if (wordFreq.size() == 0) {
            throw new IllegalStateException("IllegalStateException: wordFreqCounter has nothing. Check accumulator updating");
        }

        for (Entry<String, Double> entry : wordFreq.entrySet()) {
            String stringToken = entry.getKey();
            Double tokenCount = entry.getValue();

            // Turn words below min count to UNK
            stringToken = filterMinWord(stringToken, tokenCount);

            // Turn tokens to vocab and add to vocab cache
            addTokenToVocabCache(stringToken, tokenCount);
        }
    }

    public void buildVocabCache() {

        // Tokenize
        JavaRDD<List<String>> tokenizedRDD = tokenize();

        // Update accumulator values and map to an RDD of sentence counts
        sentenceWordsCountRDD = updateAndReturnAccumulatorVal(tokenizedRDD).cache();

        // Get value from accumulator
        Counter<String> wordFreqCounter = wordFreqAcc.value();

        // Filter out low count words and add to vocab cache object and feed into LookupCache
        filterMinWordAddVocab(wordFreqCounter);

        // At this point the vocab cache is built. Broadcast vocab cache
        vocabCacheBroadcast = sc.broadcast(vocabCache);

    }

    public void buildVocabWordListRDD() {

        if (sentenceWordsCountRDD == null)
            throw new IllegalStateException("SentenceWordCountRDD must be defined first. Run buildLookupCache first.");

        Function wordsListToVocabWords = new Function<Pair<List<String>, AtomicLong>, List<VocabWord>>() {
            @Override
            public List<VocabWord> call(Pair<List<String>, AtomicLong> pair) throws Exception {
                List<String> wordsList = pair.getFirst();
                List<VocabWord> vocabWordsList = new ArrayList<>();
                for (String s : wordsList)
                    vocabWordsList.add(vocabCacheBroadcast.getValue().wordFor(s));
                return vocabWordsList;
            }
        };

        Function getSentenceCount = new Function<Pair<List<String>, AtomicLong>, AtomicLong>() {
            @Override
            public AtomicLong call(Pair<List<String>, AtomicLong> pair) throws Exception {
                return pair.getSecond();
            }
        };
        vocabWordListRDD = sentenceWordsCountRDD.map(wordsListToVocabWords).setName("vocabWordListRDD").cache();
        sentenceCountRDD = sentenceWordsCountRDD.map(getSentenceCount).setName("sentenceCountRDD").cache();
        // Actions to fill vocabWordListRDD and sentenceCountRDD
        vocabWordListRDD.count();
        totalWordCount = sentenceCountRDD.reduce(new Function2<AtomicLong, AtomicLong, AtomicLong>() {
            @Override
            public AtomicLong call(AtomicLong a, AtomicLong b) {
                return new AtomicLong(a.get() + b.get());
            }
        }).get();

        // Release sentenceWordsCountRDD from cache
        sentenceWordsCountRDD.unpersist();
    }
}
