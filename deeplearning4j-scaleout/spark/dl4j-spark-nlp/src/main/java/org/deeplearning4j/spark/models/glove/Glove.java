/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.spark.models.glove;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.models.glove.GloveWeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.models.glove.cooccurrences.CoOccurrenceCalculator;
import org.deeplearning4j.spark.models.glove.cooccurrences.CoOccurrenceCounts;
import org.deeplearning4j.spark.text.TextPipeline;
import org.deeplearning4j.spark.text.TokenizerFunction;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * Spark glove
 *
 * @author Adam Gibson
 */
public class Glove {

    private Broadcast<VocabCache> vocabCacheBroadcast;
    private String tokenizerFactoryClazz = DefaultTokenizerFactory.class.getName();
    private boolean symmetric = true;
    private int windowSize = 15;
    private int iterations = 300;

    /**
     *
     * @param tokenizerFactoryClazz the fully qualified class name of the tokenizer
     * @param symmetric whether the co occurrence counts should be symmetric
     * @param windowSize the window size for co occurrence
     * @param iterations the number of iterations
     */
    public Glove(String tokenizerFactoryClazz, boolean symmetric, int windowSize, int iterations) {
        this.tokenizerFactoryClazz = tokenizerFactoryClazz;
        this.symmetric = symmetric;
        this.windowSize = windowSize;
        this.iterations = iterations;
    }

    /**
     *
     * @param symmetric whether the co occurrence counts should be symmetric
     * @param windowSize the window size for co occurrence
     * @param iterations the number of iterations
     */
    public Glove(boolean symmetric, int windowSize, int iterations) {
        this.symmetric = symmetric;
        this.windowSize = windowSize;
        this.iterations = iterations;
    }

    /**
     * Train on the corpus
     * @param rdd the rdd to train
     * @return the vocab and weights
     */
    public Pair<VocabCache,GloveWeightLookupTable> train(JavaRDD<String> rdd) {
        TextPipeline pipeline = new TextPipeline(rdd);
        Pair<VocabCache,Long> vocabAndNumWords = pipeline.process();
        SparkConf conf = rdd.context().getConf();
        JavaSparkContext sc = new JavaSparkContext(rdd.context());
        vocabCacheBroadcast = sc.broadcast(vocabAndNumWords.getFirst());

        GloveWeightLookupTable gloveWeightLookupTable = new GloveWeightLookupTable.Builder()
                .cache(vocabAndNumWords.getFirst()).lr(conf.getDouble(GlovePerformer.ALPHA,0.025))
                .maxCount(conf.getDouble(GlovePerformer.MAX_COUNT,100)).vectorLength(conf.getInt(GlovePerformer.VECTOR_LENGTH,100))
               .xMax(conf.getDouble(GlovePerformer.X_MAX,0.75)).build();
        gloveWeightLookupTable.resetWeights();

        CounterMap<String,String> coOccurrenceCounts = rdd.map(new TokenizerFunction(tokenizerFactoryClazz))
                .map(new CoOccurrenceCalculator(symmetric,vocabCacheBroadcast,windowSize)).fold(new CounterMap<String, String>(),new CoOccurrenceCounts());

        List<Triple<String,String,Double>> counts = new ArrayList<>();
        Iterator<Pair<String,String>> pairIter = coOccurrenceCounts.getPairIterator();
        while(pairIter.hasNext()) {
            Pair<String,String> pair = pairIter.next();
            counts.add(new Triple<>(pair.getFirst(),pair.getSecond(),coOccurrenceCounts.getCount(pair.getFirst(),pair.getSecond())));
        }

        for(int i = 0; i < iterations; i++) {
            Collections.shuffle(counts);
            JavaRDD<Triple<String,String,Double>> parallel = sc.parallelize(counts);
            JavaRDD<Triple<VocabWord,VocabWord,Double>> vocab = parallel.map(new VocabWordPairs(vocabCacheBroadcast));
            vocab.foreach(new GlovePerformer(gloveWeightLookupTable));
        }

        return new Pair<>(vocabAndNumWords.getFirst(),gloveWeightLookupTable);
    }

}
