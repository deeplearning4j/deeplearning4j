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

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.spark.models.embeddings.word2vec.Word2Vec;
import org.deeplearning4j.spark.models.embeddings.word2vec.Word2VecPerformer;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.util.Index;

import java.util.List;
import java.util.Map.Entry;

/**
 * A spark based text pipeline
 * with minimum word frequency and stop words
 *
 * @author Adam Gibson
 */
public class TextPipeline {
    private JavaRDD<String> corpus;
    private List<String> stopWords;
    private int minWordFrequency;
    public final static String MIN_WORDS = "org.deeplearning4j.spark.text.minwords";

    /**
     *
     * @param corpus the corpus of text
     * @param stopWords the stop words to use
     * @param minWordFrequency the minimum word frequency for the vocab
     */
    public TextPipeline(JavaRDD<String> corpus, List<String> stopWords, int minWordFrequency) {
        this.corpus = corpus;
        this.stopWords = stopWords;
        this.minWordFrequency = minWordFrequency;
        SparkConf conf = corpus.context().conf();
        int val = conf.getInt(MIN_WORDS,minWordFrequency);
        this.minWordFrequency = val;

    }

    /**
     * Create a text pipeline with the given corpus,
     * StopWords.getStopWords() and a minimum word frequency of 5
     * @param corpus the corpus to use
     */
    public TextPipeline(JavaRDD<String> corpus) {
        this(corpus, StopWords.getStopWords(),5);
    }

    /**
     * Create a text pipeline with the specified corpus
     * @param corpus the corpus to use
     * @param minWordFrequency the minimum word frequency to use
     */
    public TextPipeline(JavaRDD<String> corpus, int minWordFrequency) {
        this(corpus,StopWords.getStopWords(),minWordFrequency);
    }

    public Pair<VocabCache, Long> filterMinWordAddVocab(Pair<VocabCache, Long> pair) {
        // The first item is InMemoryLookupCache
        InMemoryLookupCache lookupCacheObject =  (InMemoryLookupCache)pair.getFirst();
        // The second item is word count of the corpus (reduced from word count of sentences)
        // Don't need to touch this
        Long wordCount = pair.getSecond();

        for (Entry<String, Double> entry : lookupCacheObject.getWordFrequencies().entrySet()) {
            String stringToken = entry.getKey();
            Double tokenCount = entry.getValue();
            if (tokenCount < minWordFrequency) {
                // If word frequency below min word count, it will be UNK (unknown)
                stringToken = org.deeplearning4j.models.word2vec.Word2Vec.UNK;
            }
            // Making string token into actual token if not already an actual token (vocabWord)
            VocabWord actualToken;
            if(lookupCacheObject.hasToken(stringToken))
                actualToken = lookupCacheObject.tokenFor(stringToken);
            else {
                actualToken = new VocabWord(1.0, stringToken);
            }

            // Set the index of the actual token (vocabWord)
            // Put vocabWord into vocabs in InMemoryVocabCache
            boolean vocabContainsWord = lookupCacheObject.containsWord(stringToken);
            if(!vocabContainsWord) {
                lookupCacheObject.addToken(actualToken);
                int idx = lookupCacheObject.numWords();
                actualToken.setIndex(idx);
                lookupCacheObject.putVocabWord(stringToken);
            }
        }
        return new Pair<>((VocabCache)lookupCacheObject, wordCount);
    }

    /**
     * Get a vocab cache with all of the vocab based on the
     * specified stop words and minimum word frequency
     * @param tokenizer the fully qualified class name to use for instantiating tokenizers
     * @return the vocab cache and associated total number of words
     */
    public Pair<VocabCache,Long> process(String tokenizer) {
        JavaSparkContext sc = new JavaSparkContext(corpus.context());
        Broadcast<List<String>> broadcast = sc.broadcast(stopWords);
        int nGrams = corpus.context().conf().getInt(Word2VecPerformer.N_GRAMS, 1);
        // Just getting the tokens by splitting on space, doesn't take care of punctuations
        JavaRDD<Pair<List<String>, Long>> tokenizedRDD = corpus.map(new TokenizerFunction(tokenizer, nGrams));
        Pair<VocabCache, Long> corpusCounterPair = tokenizedRDD.map(new VocabCacheFunction(new InMemoryLookupCache(), broadcast))
                .reduce(new ReduceVocabFunction());
        return filterMinWordAddVocab(corpusCounterPair);
    }

    /**
     * Get a vocab cache with all of the vocab based on the
     * specified stop words and minimum word frequency
     * @return the vocab cache and associated total number of words
     */
    public Pair<VocabCache,Long> process() {
        JavaSparkContext sc = new JavaSparkContext(corpus.context());
        Broadcast<List<String>> broadcast = sc.broadcast(stopWords);
        // Just getting the tokens by splitting on space, doesn't take care of punctuations
        JavaRDD<Pair<List<String>, Long>> tokenizedRDD = corpus.map(new TokenizerFunction(DefaultTokenizerFactory.class.getName()));
        Pair<VocabCache, Long> corpusCounterPair = tokenizedRDD.map(new VocabCacheFunction(new InMemoryLookupCache(), broadcast))
                .reduce(new ReduceVocabFunction());
        return filterMinWordAddVocab(corpusCounterPair);
    }
}
