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

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.Huffman;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.text.TextPipeline;
import org.deeplearning4j.spark.text.TokenizerFunction;
import org.deeplearning4j.spark.text.TokentoVocabWord;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Spark version of word2vec
 *
 * @author Adam Gibson
 */
public class Word2Vec implements Serializable {

    private  Broadcast<VocabCache> vocabCacheBroadcast;
    private String tokenizerFactoryClazz;
    private InMemoryLookupTable table;
    private static Logger log = LoggerFactory.getLogger(Word2Vec.class);

    public Word2Vec(String tokenizerFactoryClazz, InMemoryLookupTable table) {
        this.tokenizerFactoryClazz = tokenizerFactoryClazz;
        this.table = table;
    }

    public Word2Vec(String tokenizerFactoryClazz) {
        this.tokenizerFactoryClazz = tokenizerFactoryClazz;
    }

    public Word2Vec() {
        this(DefaultTokenizerFactory.class.getName());
    }

    /**
     * Train and return the result based on the given records.
     * Each string is assumed to be a document
     * @param rdd the rdd to train on
     * @return the vocab and lookup table for the model
     */
    public Pair<VocabCache,WeightLookupTable> train(JavaRDD<String> rdd) {
        TextPipeline pipeline = new TextPipeline(rdd);
        Pair<VocabCache,Long> vocabAndNumWords = pipeline.process(tokenizerFactoryClazz);
        SparkConf conf = rdd.context().getConf();
        final JavaSparkContext sc = new JavaSparkContext(rdd.context());
        vocabCacheBroadcast = sc.broadcast(vocabAndNumWords.getFirst());
        final InMemoryLookupTable lookupTable = this.table != null ? table : (InMemoryLookupTable) new InMemoryLookupTable.Builder()
                .cache(vocabAndNumWords.getFirst()).lr(conf.getDouble(Word2VecPerformerVoid.ALPHA,0.025))
                .vectorLength(conf.getInt(Word2VecPerformerVoid.VECTOR_LENGTH,100)).negative(conf.getDouble(Word2VecPerformerVoid.NEGATIVE,5))
                .useAdaGrad(conf.getBoolean(Word2VecPerformerVoid.ADAGRAD,false)).build();
        //only initialize if necessary
        if(this.table == null)
            lookupTable.resetWeights();

        Huffman huffman = new Huffman(vocabAndNumWords.getFirst().vocabWords());
        huffman.build();



        JavaRDD<Pair<List<VocabWord>, AtomicLong>> r = rdd
                .map(new TokenizerFunction(tokenizerFactoryClazz))
                .map(new TokentoVocabWord(vocabCacheBroadcast));

        final Word2VecParam param = new Word2VecParam.Builder()
                .negative(lookupTable.getNegative()).window(conf.getInt(Word2VecPerformer.WINDOW,5))
                .expTable(sc.broadcast(lookupTable.getExpTable())).setAlpha(lookupTable.getLr().get())
                .setMinAlpha(1e-2).setVectorLength(lookupTable.getVectorLength())
                .useAdaGrad(lookupTable.isUseAdaGrad()).weights(lookupTable)
                .build();
        param.setTotalWords(vocabAndNumWords.getSecond().intValue());


        final List<Long> wordsSeen = new ArrayList<>();
        long runningTotal = 0;
        final List<AtomicLong> frequencies = r.map(new Function<Pair<List<VocabWord>,AtomicLong>, AtomicLong>() {
            @Override
            public AtomicLong call(Pair<List<VocabWord>, AtomicLong> listAtomicLongPair) throws Exception {
                return listAtomicLongPair.getSecond();
            }
        }).collect();

        for(int j = 0; j < frequencies.size(); j++) {
            wordsSeen.add(runningTotal + frequencies.get(j).get());
            runningTotal += frequencies.get(j).get();
        }
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



        //calculate all the errors
        for(int i = 0; i < conf.getInt(Word2VecPerformerVoid.ITERATIONS,5); i++) {
            final Broadcast<Word2VecParam> finalParamBroadcast = sc.broadcast(param);
            JavaRDD<Word2VecFuncCall> call = wordsAndWordsSeen.map(new Word2VecSetup(finalParamBroadcast));
            JavaRDD<Word2VecChange> change2 = call.map(new SentenceBatch());
            change2.foreach(new VoidFunction<Word2VecChange>() {
                @Override
                public void call(Word2VecChange change) throws Exception {
                    change.apply(lookupTable);
                }
            });


            change2.unpersist();
            change2 = null;
            log.info("Iteration " + i);



        }


        return new Pair<VocabCache, WeightLookupTable>(vocabCacheBroadcast.getValue(),lookupTable);

    }


}