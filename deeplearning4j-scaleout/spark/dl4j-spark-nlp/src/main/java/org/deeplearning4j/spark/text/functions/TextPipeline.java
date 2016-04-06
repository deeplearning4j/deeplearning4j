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

package org.deeplearning4j.spark.text.functions;

import org.apache.spark.Accumulator;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.Huffman;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.spark.text.accumulators.WordFreqAccumulator;
import org.deeplearning4j.text.stopwords.StopWords;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicLong;
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
    private VocabCache<VocabWord> vocabCache = new AbstractCache<VocabWord>();
    private Broadcast<VocabCache<VocabWord>> vocabCacheBroadcast;
    private JavaRDD<List<VocabWord>> vocabWordListRDD;
    private JavaRDD<AtomicLong> sentenceCountRDD;
    private long totalWordCount;
    private boolean useUnk;
    private VectorsConfiguration configuration;

    // Empty Constructor
    public TextPipeline() {}

    // Constructor
    public TextPipeline(JavaRDD<String> corpusRDD, Broadcast<Map<String, Object>> broadcasTokenizerVarMap)
            throws Exception {
        setRDDVarMap(corpusRDD, broadcasTokenizerVarMap);
        // Setup all Spark variables
        setup();
    }

    public void setRDDVarMap(JavaRDD<String> corpusRDD,
                                     Broadcast<Map<String, Object>> broadcasTokenizerVarMap) {
        Map<String, Object> tokenizerVarMap = broadcasTokenizerVarMap.getValue();
        this.corpusRDD = corpusRDD;
        this.numWords = (int) tokenizerVarMap.get("numWords");
        // TokenizerFunction Settings
        this.nGrams = (int) tokenizerVarMap.get("nGrams");
        this.tokenizer = (String) tokenizerVarMap.get("tokenizer");
        this.tokenizerPreprocessor = (String) tokenizerVarMap.get("tokenPreprocessor");
        this.useUnk = (boolean) tokenizerVarMap.get("useUnk");
        this.configuration = (VectorsConfiguration) tokenizerVarMap.get("vectorsConfiguration");
        // Remove Stop words
       // if ((boolean) tokenizerVarMap.get("removeStop")) {
            stopWords = (List<String>) tokenizerVarMap.get("stopWords");
    //    }
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
        UpdateWordFreqAccumulatorFunction accumulatorClassFunction = new UpdateWordFreqAccumulatorFunction(stopWordBroadCast, wordFreqAcc);
        JavaRDD<Pair<List<String>, AtomicLong>> sentenceWordsCountRDD = tokenizedRDD.map(accumulatorClassFunction);

        // Loop through each element to update accumulator. Count does the same job (verified).
        sentenceWordsCountRDD.count();

        return sentenceWordsCountRDD;
    }

    private String filterMinWord(String stringToken, double tokenCount) {
        return (tokenCount < numWords) ? configuration.getUNK() : stringToken;
    }

    private void addTokenToVocabCache(String stringToken, Double tokenCount) {
        // Making string token into actual token if not already an actual token (vocabWord)
        VocabWord actualToken;
        if (vocabCache.hasToken(stringToken)) {
            actualToken = vocabCache.tokenFor(stringToken);
            actualToken.increaseElementFrequency(tokenCount.intValue());
        } else {
            actualToken = new VocabWord(tokenCount, stringToken);
        }

        // Set the index of the actual token (vocabWord)
        // Put vocabWord into vocabs in InMemoryVocabCache
        boolean vocabContainsWord = vocabCache.containsWord(stringToken);
        if (!vocabContainsWord) {
            int idx = vocabCache.numWords();

            vocabCache.addToken(actualToken);
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
            if (!useUnk && stringToken.equals("UNK")) {
                // Turn tokens to vocab and add to vocab cache
            } else addTokenToVocabCache(stringToken, tokenCount);
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

        // huffman tree should be built BEFORE vocab broadcast
        Huffman huffman = new Huffman(vocabCache.vocabWords());
        huffman.build();
        huffman.applyIndexes(vocabCache);

        // At this point the vocab cache is built. Broadcast vocab cache
        vocabCacheBroadcast = sc.broadcast(vocabCache);

    }

    public void buildVocabWordListRDD() {

        if (sentenceWordsCountRDD == null)
            throw new IllegalStateException("SentenceWordCountRDD must be defined first. Run buildLookupCache first.");

        vocabWordListRDD = sentenceWordsCountRDD.map(new WordsListToVocabWordsFunction(vocabCacheBroadcast))
                .setName("vocabWordListRDD").cache();
        sentenceCountRDD = sentenceWordsCountRDD.map(new GetSentenceCountFunction())
                .setName("sentenceCountRDD").cache();
        // Actions to fill vocabWordListRDD and sentenceCountRDD
        vocabWordListRDD.count();
        totalWordCount = sentenceCountRDD.reduce(new ReduceSentenceCount()).get();

        // Release sentenceWordsCountRDD from cache
        sentenceWordsCountRDD.unpersist();
    }

    // Getters
    public Accumulator<Counter<String>> getWordFreqAcc() {
        if (wordFreqAcc != null) {
            return wordFreqAcc;
        } else {
            throw new IllegalStateException("IllegalStateException: wordFreqAcc not set at TextPipline.");
        }
    }

    public Broadcast<VocabCache<VocabWord>> getBroadCastVocabCache() throws IllegalStateException {
        if (vocabCache.numWords() > 0) {
            return vocabCacheBroadcast;
        } else {
            throw new IllegalStateException("IllegalStateException: VocabCache not set at TextPipline.");
        }
    }

    public VocabCache<VocabWord> getVocabCache() throws IllegalStateException {
        if (vocabCache != null && vocabCache.numWords() > 0) {
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

    public JavaRDD<List<VocabWord>> getVocabWordListRDD() throws IllegalStateException {
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
}
