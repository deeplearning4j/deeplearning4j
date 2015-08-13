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

package org.deeplearning4j.spark.models.embeddings.word2vec;

import lombok.NoArgsConstructor;
import org.apache.spark.Accumulator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.word2vec.Huffman;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.text.*;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import static org.deeplearning4j.spark.models.embeddings.word2vec.Word2VecVariables.*;

/**
 * Spark version of word2vec
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class Word2Vec extends WordVectorsImpl implements Serializable  {

    private InMemoryLookupTable table;
    private static Logger log = LoggerFactory.getLogger(Word2Vec.class);

    // Constructor to take InMemoryLookupCache table from an already trained model
    public Word2Vec(InMemoryLookupTable table) { this.table = table; }


    // Training word2vec based on corpus
    public Pair<VocabCache,WeightLookupTable> train(JavaRDD<String> corpusRDD) throws Exception {

        // Each `train()` can use different parameters
        final JavaSparkContext sc = new JavaSparkContext(corpusRDD.context());
        final SparkConf conf = sc.getConf();
        int vectorLength = assignVar(VECTOR_LENGTH, conf, Integer.class);
        boolean useAdaGrad = assignVar(ADAGRAD, conf, Boolean.class);
        double negative = assignVar(NEGATIVE, conf, Double.class);
        int numWords = assignVar(NUM_WORDS, conf, Integer.class);
        int window = assignVar(WINDOW, conf, Integer.class);
        double alpha = assignVar(ALPHA, conf, Double.class);
        double minAlpha = assignVar(MIN_ALPHA, conf, Double.class);
        int iterations = assignVar(ITERATIONS, conf, Integer.class);
        int nGrams = assignVar(N_GRAMS, conf, Integer.class);
        String tokenizer = assignVar(TOKENIZER, conf, String.class);
        String tokenPreprocessor = assignVar(TOKEN_PREPROCESSOR, conf, String.class);
        boolean removeStop = assignVar(REMOVE_STOPWORDS, conf, Boolean.class);

        // Variables to fill in in train
        final JavaRDD<AtomicLong> sentenceWordsCountRDD;
        final JavaRDD<List<VocabWord>> vocabWordListRDD;
        final Long totalWordCount;
        final JavaPairRDD<List<VocabWord>, Long> vocabWordListSentenceCumSumRDD;
        final VocabCache vocabCache;
        final Broadcast<VocabCache> vocabCacheBroadcast;
        final JavaRDD<Long> sentenceCumSumCountRDD;

        // Start Training //
        //////////////////////////////////////
        log.info("Tokenization and building VocabCache ...");
        // Processing every sentence and make a VocabCache which gets fed into a LookupCache
        TextPipeline pipeline = new TextPipeline(corpusRDD, numWords, nGrams, tokenizer,
                                                 tokenPreprocessor, removeStop);
        pipeline.buildVocabCache();

        // Get total word count
        totalWordCount = pipeline.getTotalWordCount();

        // 2 RDDs: (vocab words list) and (sentence Count).Already cached
        sentenceWordsCountRDD = pipeline.getSentenceCountRDD();
        vocabWordListRDD = pipeline.getvocabWordListRDD();

        // Get vocabCache and broad-casted vocabCache
        vocabCache = pipeline.getVocabCache();
        vocabCacheBroadcast = pipeline.getBroadCastVocabCache();

        //////////////////////////////////////
        log.info("Building Huffman Tree ...");
        // Building Huffman Tree would update the code and point in each of the vocabWord in vocabCache
        Huffman huffman = new Huffman(vocabCache.vocabWords());
        huffman.build();

        //////////////////////////////////////
        log.info("Calculating cumulative sum of sentence counts ...");
        sentenceCumSumCountRDD =  new CountCumSum(sentenceWordsCountRDD).buildCumSum();

        //////////////////////////////////////
        log.info("Mapping to RDD(vocabWordList, cumulative sentence count) ...");
        vocabWordListSentenceCumSumRDD = vocabWordListRDD.zip(sentenceCumSumCountRDD);

        //////////////////////////////////////
        log.info("Building Word2VecParams ...");
        // Define InMemoryLookupTable with InMemoryLookupCache
        final InMemoryLookupTable lookupTable;
        if (table != null) {
            lookupTable = table;
        } else {
            lookupTable = (InMemoryLookupTable) new InMemoryLookupTable.Builder()
                    .cache(vocabCache).vectorLength(vectorLength).negative(negative).build();
            lookupTable.resetWeights();
        }
        //Define Word2VecParam from InMemoryLookupTable
        final Word2VecParam param = new Word2VecParam.Builder()
                .negative(negative).window(window)
                .expTable(sc.broadcast(lookupTable.getExpTable()))
                .setAlpha(alpha)
                .setMinAlpha(minAlpha)
                .setVectorLength(vectorLength)
                .useAdaGrad(useAdaGrad)
                .weights(lookupTable)
                .build();

        param.setTotalWords(totalWordCount.intValue());


        ////////////////////////////////////
        log.info("Calculating word frequencies...");


        JavaRDD<List<VocabWord>> words = r.map(new Function<Pair<List<VocabWord>, AtomicLong>, List<VocabWord>>() {
            @Override
            public List<VocabWord> call(Pair<List<VocabWord>, AtomicLong> listAtomicLongPair) throws Exception {
                return listAtomicLongPair.getFirst();
            }
        });

        JavaPairRDD<List<VocabWord>,Long> wordsAndWordsSeen = words.zipWithIndex().mapToPair(new PairFunction<Tuple2<List<VocabWord>, Long>, List<VocabWord>, Long>() {
            @Override
            public Tuple2<List<VocabWord>, Long> call(Tuple2<List<VocabWord>, Long> listLongTuple2) throws Exception {
                return new Tuple2<>(listLongTuple2._1(),wordsSeen.get(listLongTuple2._2().intValue()));
            }
        }).cache();


        log.info("Training word 2vec");
        //calculate all the errors
        for(int i = 0; i < conf.getInt(Word2VecPerformerVoid.ITERATIONS,5); i++) {
            final Broadcast<Word2VecParam> finalParamBroadcast = sc.broadcast(param);
            if(finalParamBroadcast.value() == null)
                throw new IllegalStateException("Value not found for param broadcast");
            JavaRDD<Word2VecFuncCall> call = wordsAndWordsSeen.map(new Word2VecSetup(finalParamBroadcast));
            JavaRDD<Word2VecChange> change2 = call.map(new SentenceBatch());
            change2.foreach(new VoidFunction<Word2VecChange>() {
                @Override
                public void call(Word2VecChange change) throws Exception {
                    change.apply(lookupTable);
                }
            });

            change2.unpersist();
            log.info("Iteration " + i);
        }

        // For WordVectorImpl, so nearestWord can be called
        super.lookupTable = lookupTable;
        super.vocab = vocabCacheBroadcast.getValue();

        return new Pair<VocabCache, WeightLookupTable>(vocabCacheBroadcast.getValue(),lookupTable);

    }

    public Broadcast<VocabCache> getVocabCacheBroadcast() {
        return vocabCacheBroadcast;
    }

    public void setVocabCacheBroadcast(Broadcast<VocabCache> vocabCacheBroadcast) {
        this.vocabCacheBroadcast = vocabCacheBroadcast;
    }

    public String getTokenizerFactoryClazz() {
        return tokenizerFactoryClazz;
    }

    public void setTokenizerFactoryClazz(String tokenizerFactoryClazz) {
        this.tokenizerFactoryClazz = tokenizerFactoryClazz;
    }

    public InMemoryLookupTable getTable() {
        return table;
    }

    public void setTable(InMemoryLookupTable table) {
        this.table = table;
    }
}