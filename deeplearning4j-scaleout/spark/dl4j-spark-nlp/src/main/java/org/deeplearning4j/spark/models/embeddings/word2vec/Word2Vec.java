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
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.word2vec.Huffman;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.text.functions.CountCumSum;
import org.deeplearning4j.spark.text.functions.TextPipeline;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Spark version of word2vec
 *
 * @author Adam Gibson
 */
public class Word2Vec extends WordVectorsImpl implements Serializable  {

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
    private double minAlpha = 1e-2;
    private int iterations = 1;
    private int nGrams = 1;
    private String tokenizer = "org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory";
    private String tokenPreprocessor = "org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor";
    private boolean removeStop = false;
    private long seed = 42L;

//    // Constructor to take InMemoryLookupCache table from an already trained model
//    public Word2Vec(INDArray trainedSyn1) { this.trainedSyn1 = trainedSyn1; }

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
        }};
    }

    // Training word2vec based on corpus
    public Pair<VocabCache,WeightLookupTable> train(JavaRDD<String> corpusRDD) throws Exception {
        log.info("Start training ...");

        // Each `train()` can use different parameters
        final JavaSparkContext sc = new JavaSparkContext(corpusRDD.context());

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
        TextPipeline pipeline = new TextPipeline(corpusRDD, getTokenizerVarMap());
        pipeline.buildVocabCache();
        pipeline.buildVocabWordListRDD();

        // Get total word count
        totalWordCount = pipeline.getTotalWordCount();

        // 2 RDDs: (vocab words list) and (sentence Count).Already cached
        sentenceWordsCountRDD = pipeline.getSentenceCountRDD();
        vocabWordListRDD = pipeline.getVocabWordListRDD();

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
        vocabWordListSentenceCumSumRDD = vocabWordListRDD.zip(sentenceCumSumCountRDD)
                .setName("vocabWordListSentenceCumSumRDD").cache();

        /////////////////////////////////////
        log.info("Broadcasting word2vec variable map to workers ...");
        Broadcast<Map<String, Object>> word2vecVarMapBroadcast = sc.broadcast(getWord2vecVarMap());

        /////////////////////////////////////
//        vocabWordListSentenceCumSumRDD.map(new FirstIterationFunction(word2vecVarMapBroadcast))





        //////////////////////////////////////
//        log.info("Building Word2VecParams ...");
//        // Define InMemoryLookupTable with InMemoryLookupCache
//        final INDArray syn1;
//        if (trainedSyn1 != null) {
//            syn1 = trainedSyn1;
//        } else {
//            lookupTable = (InMemoryLookupTable) new InMemoryLookupTable.Builder()
//                    .cache(vocabCache).vectorLength(vectorLength).negative(negative).build();
//            lookupTable.resetWeights();
//        }
//        //Define Word2VecParam from InMemoryLookupTable
//        final Word2VecParam param = new Word2VecParam.Builder()
//                .negative(negative).window(window)
//                .expTable(sc.broadcast(lookupTable.getExpTable()))
//                .setAlpha(alpha)
//                .setMinAlpha(minAlpha)
//                .setVectorLength(vectorLength)
//                .useAdaGrad(useAdaGrad)
//                .weights(lookupTable)
//                .build();
//        param.setTotalWords(totalWordCount.intValue());

        //////////////////////////////////////
//        log.info("Training word2vec");
//        //calculate all the errors
//        Word2VecParam prevParam;
//        for(int i = 0; i < iterations; i++) {
//            final Broadcast<Word2VecParam> finalParamBroadcast = sc.broadcast(param);
//
//            if(finalParamBroadcast.value() == null)
//                throw new IllegalStateException("Value not found for param broadcast");
//            JavaRDD<Word2VecFuncCall> call = vocabWordListSentenceCumSumRDD.map(new Word2VecSetup(finalParamBroadcast));
//            JavaRDD<Word2VecChange> change2 = call.map(new SentenceBatch());
//            change2.foreach(new VoidFunction<Word2VecChange>() {
//                @Override
//                public void call(Word2VecChange change) throws Exception {
//                    change.apply(lookupTable);
//                }
//            });
//
//            change2.unpersist();
//            prevParam = param;
//            // TODO: Have to send the changes back and build logic to apply changes to lookuptable and broadcast again
//            if (prevParam.equals(param)) {
//                throw new Exception("Param is not updated.");
//            }
//            log.info("Iteration " + i);
//        }
//
//        // For WordVectorImpl, so nearestWord can be called
//        super.lookupTable = lookupTable;
//        super.vocab = vocabCacheBroadcast.getValue();
//
        return new Pair<VocabCache, WeightLookupTable>(vocabCacheBroadcast.getValue(),lookupTable);
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

}