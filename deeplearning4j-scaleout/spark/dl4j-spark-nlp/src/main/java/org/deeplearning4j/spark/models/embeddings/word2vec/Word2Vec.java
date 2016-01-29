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

import org.apache.commons.math3.util.FastMath;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.embeddings.reader.impl.FlatModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.word2vec.Huffman;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.text.functions.CountCumSum;
import org.deeplearning4j.spark.text.functions.TextPipeline;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Spark version of word2vec
 *
 * @author Adam Gibson
 */
@Deprecated
public class Word2Vec extends WordVectorsImpl<VocabWord> implements Serializable  {

    private INDArray trainedSyn1;
    private static Logger log = LoggerFactory.getLogger(Word2Vec.class);
    private int MAX_EXP = 6;
    private double[] expTable;

    // Input by user only via setters
    private int vectorLength = 100;
    private boolean useAdaGrad = false;
    private int negative = 0;
    private int numWords = 1;
    private int window = 5;
    private double alpha= 0.025;
    private double minAlpha = 0.0001;
    private int iterations = 1;
    private int nGrams = 1;
    private String tokenizer = "org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory";
    private String tokenPreprocessor = "org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor";
    private boolean removeStop = false;
    private long seed = 42L;

    // Constructor to take InMemoryLookupCache table from an already trained model
    public Word2Vec(INDArray trainedSyn1) {
        this.trainedSyn1 = trainedSyn1;
        this.expTable = initExpTable();
    }

    public Word2Vec() {
        this.expTable = initExpTable();
    }

    public double[] initExpTable() {
        double[] expTable = new double[1000];
        for (int i = 0; i < expTable.length; i++) {
            double tmp = FastMath.exp((i / (double) expTable.length * 2 - 1) * MAX_EXP);
            expTable[i] = tmp / (tmp + 1.0);
        }
        return expTable;
    }

    public Map<String, Object> getTokenizerVarMap() {
        return new HashMap<String, Object>() {{
            put("numWords", numWords);
            put("nGrams", nGrams);
            put("tokenizer", tokenizer);
            put("tokenPreprocessor", tokenPreprocessor);
            put("removeStop", removeStop);
        }};
    }

    public Map<String, Object> getWord2vecVarMap() {
        return new HashMap<String, Object>() {{
            put("vectorLength", vectorLength);
            put("useAdaGrad", useAdaGrad);
            put("negative", negative);
            put("window", window);
            put("alpha", alpha);
            put("minAlpha", minAlpha);
            put("iterations", iterations);
            put("seed", seed);
            put("maxExp", MAX_EXP);
        }};
    }

    // Training word2vec based on corpus
    public void train(JavaRDD<String> corpusRDD) throws Exception {
        log.info("Start training ...");

        // SparkContext
        final JavaSparkContext sc = new JavaSparkContext(corpusRDD.context());

        // Pre-defined variables
        Map<String, Object> tokenizerVarMap = getTokenizerVarMap();
        Map<String, Object> word2vecVarMap = getWord2vecVarMap();

        // Variables to fill in in train
        final JavaRDD<AtomicLong> sentenceWordsCountRDD;
        final JavaRDD<List<VocabWord>> vocabWordListRDD;
        final JavaPairRDD<List<VocabWord>, Long> vocabWordListSentenceCumSumRDD;
        final VocabCache<VocabWord> vocabCache;
        final JavaRDD<Long> sentenceCumSumCountRDD;

        // Start Training //
        //////////////////////////////////////
        log.info("Tokenization and building VocabCache ...");
        // Processing every sentence and make a VocabCache which gets fed into a LookupCache
        Broadcast<Map<String, Object>> broadcastTokenizerVarMap = sc.broadcast(tokenizerVarMap);
        TextPipeline pipeline = new TextPipeline(corpusRDD, broadcastTokenizerVarMap);
        pipeline.buildVocabCache();
        pipeline.buildVocabWordListRDD();

        // Get total word count and put into word2vec variable map
        word2vecVarMap.put("totalWordCount", pipeline.getTotalWordCount());

        // 2 RDDs: (vocab words list) and (sentence Count).Already cached
        sentenceWordsCountRDD = pipeline.getSentenceCountRDD();
        vocabWordListRDD = pipeline.getVocabWordListRDD();

        // Get vocabCache and broad-casted vocabCache
        Broadcast<VocabCache<VocabWord>> vocabCacheBroadcast = pipeline.getBroadCastVocabCache();
        vocabCache = vocabCacheBroadcast.getValue();

        //////////////////////////////////////
        log.info("Building Huffman Tree ...");
        // Building Huffman Tree would update the code and point in each of the vocabWord in vocabCache
/*        Huffman huffman = new Huffman(vocabCache.vocabWords());
        huffman.build();
        huffman.applyIndexes(vocabCache);
*/
        //////////////////////////////////////
        log.info("Calculating cumulative sum of sentence counts ...");
        sentenceCumSumCountRDD =  new CountCumSum(sentenceWordsCountRDD).buildCumSum();

        //////////////////////////////////////
        log.info("Mapping to RDD(vocabWordList, cumulative sentence count) ...");
        vocabWordListSentenceCumSumRDD = vocabWordListRDD.zip(sentenceCumSumCountRDD)
                .setName("vocabWordListSentenceCumSumRDD").cache();

        /////////////////////////////////////
        log.info("Broadcasting word2vec variables to workers ...");
        Broadcast<Map<String, Object>> word2vecVarMapBroadcast = sc.broadcast(word2vecVarMap);
        Broadcast<double[]> expTableBroadcast = sc.broadcast(expTable);



        /////////////////////////////////////
        log.info("Training word2vec sentences ...");
        FlatMapFunction firstIterFunc = new FirstIterationFunction(word2vecVarMapBroadcast, expTableBroadcast);
        @SuppressWarnings("unchecked")
        JavaRDD< Pair<Integer, INDArray> > indexSyn0UpdateEntryRDD =
                vocabWordListSentenceCumSumRDD.mapPartitions(firstIterFunc)
                .map(new MapToPairFunction());

        // Get all the syn0 updates into a list in driver
        List<Pair<Integer, INDArray>> syn0UpdateEntries = indexSyn0UpdateEntryRDD.collect();

        // Instantiate syn0
        INDArray syn0 = Nd4j.zeros(vocabCache.numWords(), vectorLength);

        // Updating syn0
        int cnt = 0;
        Map<Integer, AtomicInteger> updates = new HashMap<>();
        for (Pair<Integer, INDArray> syn0UpdateEntry : syn0UpdateEntries) {
            if (syn0UpdateEntry.getFirst().equals(new Integer(126))) {
                log.info("Two before transfer: " + syn0UpdateEntry.getSecond());
            }
            syn0.getRow(syn0UpdateEntry.getFirst()).addi(syn0UpdateEntry.getSecond());
            if (syn0UpdateEntry.getFirst().equals(new Integer(126))) {
                log.info("Two after transfer: " + syn0.getRow(syn0UpdateEntry.getFirst()));
            }
            cnt++;
            if (updates.containsKey(syn0UpdateEntry.getFirst())) {
                updates.get(syn0UpdateEntry.getFirst()).incrementAndGet();
            } else updates.put(syn0UpdateEntry.getFirst(), new AtomicInteger(1));
        }
        log.info("Total entries transferred: " + cnt);

        cnt = 0;
        for (Map.Entry<Integer, AtomicInteger> entry: updates.entrySet()) {
            if (entry.getValue().get() > 1) {
                syn0.getRow(entry.getKey()).divi(entry.getValue().get());
                cnt++;
            }
        }
        log.info("Total entries merged: " + cnt);

        vocab = vocabCache;
        InMemoryLookupTable<VocabWord> inMemoryLookupTable = new InMemoryLookupTable<VocabWord>();
        inMemoryLookupTable.setVocab(vocabCache);
        inMemoryLookupTable.setVectorLength(vectorLength);
        inMemoryLookupTable.setSyn0(syn0);
        lookupTable = inMemoryLookupTable;
        modelUtils.init(lookupTable);
    }

    public int getVectorLength() {
        return vectorLength;
    }

    public Word2Vec setVectorLength(int vectorLength) {
        this.vectorLength = vectorLength;
        return this;
    }

    public boolean isUseAdaGrad() {
        return useAdaGrad;
    }

    public Word2Vec setUseAdaGrad(boolean useAdaGrad) {
        this.useAdaGrad = useAdaGrad;
        return this;
    }

    public int getNegative() {
        return negative;
    }

    public Word2Vec setNegative(int negative) {
        this.negative = negative;
        return this;
    }

    public int getNumWords() {
        return numWords;
    }

    public Word2Vec setNumWords(int numWords) {
        this.numWords = numWords;
        return this;
    }

    public int getWindow() {
        return window;
    }

    public Word2Vec setWindow(int window) {
        this.window = window;
        return this;
    }

    public double getAlpha() {
        return alpha;
    }

    public Word2Vec setAlpha(double alpha) {
        this.alpha = alpha;
        return this;
    }

    public double getMinAlpha() {
        return minAlpha;
    }

    public Word2Vec setMinAlpha(double minAlpha) {
        this.minAlpha = minAlpha;
        return this;
    }

    public int getIterations() {
        return iterations;
    }

    public Word2Vec setIterations(int iterations) {
        this.iterations = iterations;
        return this;
    }

    public int getnGrams() {
        return nGrams;
    }

    public Word2Vec setnGrams(int nGrams) {
        this.nGrams = nGrams;
        return this;
    }

    public String getTokenizer() {
        return tokenizer;
    }

    public Word2Vec setTokenizer(String tokenizer) {
        this.tokenizer = tokenizer;
        return this;
    }

    public String getTokenPreprocessor() {
        return tokenPreprocessor;
    }

    public Word2Vec setTokenPreprocessor(String tokenPreprocessor) {
        this.tokenPreprocessor = tokenPreprocessor;
        return this;
    }

    public boolean isRemoveStop() {
        return removeStop;
    }

    public Word2Vec setRemoveStop(boolean removeStop) {
        this.removeStop = removeStop;
        return this;
    }

    public long getSeed() {
        return seed;
    }

    public Word2Vec setSeed(long seed) {
        this.seed = seed;
        return this;
    }

    public double[] getExpTable() {
        return expTable;
    }
}