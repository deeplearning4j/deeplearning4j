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

import org.apache.spark.Accumulator;
import org.apache.spark.AccumulatorParam;
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
import org.deeplearning4j.spark.text.MaxPerPartitionAccumulator;
import org.deeplearning4j.spark.text.TextPipeline;
import org.deeplearning4j.spark.text.TokenizerFunction;
import org.deeplearning4j.spark.text.TokentoVocabWord;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Spark version of word2vec
 *
 * @author Adam Gibson
 */
public class Word2Vec extends WordVectorsImpl implements Serializable  {

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
        SparkConf conf = rdd.context().getConf();
        int minWords = conf.getInt(Word2VecPerformer.NUM_WORDS, 5);
        TextPipeline pipeline = new TextPipeline(rdd, minWords);
        Pair<VocabCache,Long> vocabAndNumWords = pipeline.process(tokenizerFactoryClazz);
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

        log.info("Built huffman tree");

        JavaRDD<Pair<List<VocabWord>, AtomicLong>> r = rdd
                .map(new TokenizerFunction(tokenizerFactoryClazz))
                .map(new TokentoVocabWord(vocabCacheBroadcast));

        log.info("Built vocab..");
        final Word2VecParam param = new Word2VecParam.Builder()
                .negative(0.0).window(conf.getInt(Word2VecPerformer.WINDOW,5))
                .expTable(sc.broadcast(lookupTable.getExpTable())).setAlpha(lookupTable.getLr().get())
                .setMinAlpha(1e-2).setVectorLength(lookupTable.getVectorLength())
                .useAdaGrad(lookupTable.isUseAdaGrad()).weights(lookupTable)
                .build();

        param.getTotalWords();
        param.setTotalWords(vocabAndNumWords.getSecond().intValue());

        log.info("Counting words within sentences..");
        // Get all the frequencies
        final JavaRDD<AtomicLong> frequencies = r.map(new Function<Pair<List<VocabWord>, AtomicLong>, AtomicLong>() {
            @Override
            public AtomicLong call(Pair<List<VocabWord>, AtomicLong> listAtomicLongPair) throws Exception {
                return listAtomicLongPair.getSecond();
            }
        }).cache();

        // Accumulator to get the max of the cumulative sum in each partition
        final Accumulator<Counter<Integer>> maxPerPartitionAcc = sc.accumulator(new Counter<Integer>(),
                new MaxPerPartitionAccumulator());

        //Do a scan-left equivalent in each partition
        Function2 foldWithinPartition = new Function2<Integer, Iterator<AtomicLong>, Iterator<Long>>(){
            @Override
            public Iterator<Long> call(Integer ind, Iterator<AtomicLong> partition) throws Exception {

                List<Long> foldedItemList = new ArrayList<Long>() {{ add(0L); }};

                while (partition.hasNext()) {
                    Long curPartitionItem = partition.next().get();
                    Integer lastFoldedIndex = foldedItemList.size() - 1;
                    Long lastFoldedItem = foldedItemList.get(lastFoldedIndex);
                    Long sumLastCurrent = curPartitionItem + lastFoldedItem;

                    foldedItemList.set(lastFoldedIndex, sumLastCurrent);
                    foldedItemList.add(sumLastCurrent);
                }

                // Update Accumulator
                Long maxFoldedItem = foldedItemList.remove(foldedItemList.size() - 1);
                Counter<Integer> partitionIndex2maxItemCounter = new Counter<>();
                partitionIndex2maxItemCounter.incrementCount(ind, maxFoldedItem);
                maxPerPartitionAcc.add(partitionIndex2maxItemCounter);

                return foldedItemList.iterator();
            }
        };

        // Partition mapping to fold within partition
        @SuppressWarnings("unchecked")
        JavaRDD<Long> foldWithinPartitionRDD = frequencies.mapPartitionsWithIndex(foldWithinPartition, true);
        // Action to fill the accumulator
        foldWithinPartitionRDD.foreachPartition(new VoidFunction<Iterator<Long>>() {
            @Override
            public void call(Iterator<Long> integerIterator) throws Exception {
            }
        });
        //Cache
        foldWithinPartitionRDD.cache();

        // Get the max count of the cumulative within each partition from accumulator
        final Counter<Integer> maxPerPartitionCounter = maxPerPartitionAcc.value();
        // Broadcast max count of cumulative of each partition
        final Broadcast<Counter<Integer>> broadcastedmaxPerPartitionCounter = sc.broadcast(maxPerPartitionCounter);

        // Fold between partitions based on max count of cumulative of each partition
        Function2 foldBetweenPartitions = new Function2<Integer, Iterator<Long>, Iterator<Long>>() {
            @Override
            public Iterator<Long> call(Integer ind, Iterator<Long> partition) throws Exception {
                int sumToAdd = 0;
                Counter<Integer> maxPerPartitionCounterInScope = broadcastedmaxPerPartitionCounter.value();
                if (ind != 0) {
                    for (int i=0; i < ind; i++) { sumToAdd += maxPerPartitionCounterInScope.getCount(i); }
                }

                List<Long> itemsAddedToList = new ArrayList<>();
                while (partition.hasNext()) {
                    itemsAddedToList.add(partition.next() + sumToAdd);
                }

                return itemsAddedToList.iterator();
            }
        };
        @SuppressWarnings("unchecked")
        JavaRDD foldBetweenPartitionRDD = foldWithinPartitionRDD.mapPartitionsWithIndex(foldBetweenPartitions, true);
        @SuppressWarnings("unchecked")
        final List<Long> wordsSeen = foldBetweenPartitionRDD.collect();


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