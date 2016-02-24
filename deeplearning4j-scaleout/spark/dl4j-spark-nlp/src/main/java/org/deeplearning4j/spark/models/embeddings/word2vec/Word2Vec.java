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

import lombok.Getter;
import lombok.NonNull;
import org.apache.commons.math3.util.FastMath;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.embeddings.reader.impl.FlatModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.word2vec.Huffman;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.text.functions.CountCumSum;
import org.deeplearning4j.spark.text.functions.TextPipeline;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Spark version of word2vec
 *
 * @author Adam Gibson
 * @author raver119@gmail.com
 */
public class Word2Vec extends WordVectorsImpl<VocabWord> implements Serializable  {

    private INDArray trainedSyn1;
    private static Logger log = LoggerFactory.getLogger(Word2Vec.class);
    private int MAX_EXP = 6;
    @Getter private double[] expTable;
    @Getter protected VectorsConfiguration configuration;

    // Input by user only via setters
    private int nGrams = 1;
    private String tokenizer = "org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory";
    private String tokenPreprocessor = "org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor";
    private boolean removeStop = false;
    private long seed = 42L;
    private boolean useUnknown = false;

    // Constructor to take InMemoryLookupCache table from an already trained model
    protected Word2Vec(INDArray trainedSyn1) {
        this.trainedSyn1 = trainedSyn1;
        this.expTable = initExpTable();
    }

    protected Word2Vec() {
        this.expTable = initExpTable();
    }

    protected double[] initExpTable() {
        double[] expTable = new double[100000];
        for (int i = 0; i < expTable.length; i++) {
            double tmp = FastMath.exp((i / (double) expTable.length * 2 - 1) * MAX_EXP);
            expTable[i] = tmp / (tmp + 1.0);
        }
        return expTable;
    }

    public Map<String, Object> getTokenizerVarMap() {
        return new HashMap<String, Object>() {{
            put("numWords", minWordFrequency);
            put("nGrams", nGrams);
            put("tokenizer", tokenizer);
            put("tokenPreprocessor", tokenPreprocessor);
            put("removeStop", removeStop);
            put("stopWords", stopWords);
            put("useUnk", useUnknown);
        }};
    }

    public Map<String, Object> getWord2vecVarMap() {
        return new HashMap<String, Object>() {{
            put("vectorLength", layerSize);
            put("useAdaGrad", useAdeGrad);
            put("negative", negative);
            put("window", window);
            put("alpha", learningRate.get());
            put("minAlpha", minLearningRate);
            put("iterations", numIterations);
            put("seed", seed);
            put("maxExp", MAX_EXP);
            put("batchSize", batchSize);
        }};
    }

    /**
     *  Training word2vec model on a given text corpus
     *
     * @param corpusRDD training corpus
     * @throws Exception
     */
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
/*
        We don't need to build tree here, since it was built earlier, at TextPipeline.buildVocabCache() call.

        Huffman huffman = new Huffman(vocabCache.vocabWords());
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
        FlatMapFunction firstIterFunc = new FirstIterationFunction(word2vecVarMapBroadcast, expTableBroadcast, vocabCacheBroadcast);
        @SuppressWarnings("unchecked")
        JavaRDD< Pair<Integer, INDArray> > indexSyn0UpdateEntryRDD =
                vocabWordListSentenceCumSumRDD.mapPartitions(firstIterFunc)
                .map(new MapToPairFunction());

        // Get all the syn0 updates into a list in driver
        List<Pair<Integer, INDArray>> syn0UpdateEntries = indexSyn0UpdateEntryRDD.collect();

        // Instantiate syn0
        INDArray syn0 = Nd4j.zeros(vocabCache.numWords(), layerSize);

        // Updating syn0 first pass: just add vectors obtained from different nodes
        Map<Integer, AtomicInteger> updates = new HashMap<>();
        for (Pair<Integer, INDArray> syn0UpdateEntry : syn0UpdateEntries) {
            syn0.getRow(syn0UpdateEntry.getFirst()).addi(syn0UpdateEntry.getSecond());

            // for proper averaging we need to divide resulting sums later, by the number of additions
            if (updates.containsKey(syn0UpdateEntry.getFirst())) {
                updates.get(syn0UpdateEntry.getFirst()).incrementAndGet();
            } else updates.put(syn0UpdateEntry.getFirst(), new AtomicInteger(1));
        }

        // Updating syn0 second pass: average obtained vectors
        for (Map.Entry<Integer, AtomicInteger> entry: updates.entrySet()) {
            if (entry.getValue().get() > 1) {
                syn0.getRow(entry.getKey()).divi(entry.getValue().get());
            }
        }


        vocab = vocabCache;
        InMemoryLookupTable<VocabWord> inMemoryLookupTable = new InMemoryLookupTable<VocabWord>();
        inMemoryLookupTable.setVocab(vocabCache);
        inMemoryLookupTable.setVectorLength(layerSize);
        inMemoryLookupTable.setSyn0(syn0);
        lookupTable = inMemoryLookupTable;
        modelUtils.init(lookupTable);
    }



    public static class Builder {
        protected int nGrams = 1;
        protected int numIterations = 1;
        protected int minWordFrequency = 1;
        protected int numEpochs = 1;
        protected double learningRate = 0.025;
        protected double minLearningRate = 0.001;
        protected int windowSize = 5;
        protected double negative = 0;
        protected double sampling = 1e-5;
        protected long seed = 42L;
        protected boolean useAdaGrad = false;
        protected TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        protected VectorsConfiguration configuration = new VectorsConfiguration();
        protected int layerSize;
        protected List<String> stopWords = new ArrayList<>();
        protected int batchSize = 100;
        protected boolean useUnk = false;
        private String tokenizer = "";
        private String tokenPreprocessor = "";

        /**
         * Creates Builder instance with default parameters set.
         */
        public Builder() {
            this(new VectorsConfiguration());
        }

        /**
         * Uses VectorsConfiguration bean to initialize Word2Vec model parameters
         *
         * @param configuration
         */
        public Builder(VectorsConfiguration configuration) {
            this.configuration = configuration;
            this.numIterations = configuration.getIterations();
            this.numEpochs = configuration.getEpochs();
            this.minLearningRate = configuration.getMinLearningRate();
            this.learningRate = configuration.getLearningRate();
            this.sampling = configuration.getSampling();
            this.negative = configuration.getNegative();
            this.minWordFrequency = configuration.getMinWordFrequency();
            this.seed = configuration.getSeed();
//            this.stopWords = configuration.get

            //  TODO: investigate this
            //this.hugeModelExpected = configuration.isHugeModelExpected();

            this.batchSize = configuration.getBatchSize();
            this.layerSize = configuration.getLayersSize();

          //  this.learningRateDecayWords = configuration.getLearningRateDecayWords();
            this.useAdaGrad = configuration.isUseAdaGrad();
            this.windowSize = configuration.getWindow();

            if (configuration.getStopList() != null) this.stopWords.addAll(configuration.getStopList());
        }

        /**
         * Specifies window size
         *
         * @param windowSize
         * @return
         */
        public Builder windowSize(int windowSize) {
            this.windowSize = windowSize;
            return this;
        }

        /**
         * Specifies negative sampling
         * @param negative
         * @return
         */
        public Builder negative(int negative) {
            this.negative = negative;
            return this;
        }

        /**
         * Specifies subsamplng value
         *
         * @param sampling
         * @return
         */
        public Builder sampling(double sampling) {
            this.sampling = sampling;
            return this;
        }

        /**
         * This method specifies initial learning rate for model
         *
         * @param lr
         * @return
         */
        public Builder learningRate(double lr) {
            this.learningRate = lr;
            return this;
        }

        /**
         * This method specifies bottom threshold for learning rate decay
         *
         * @param mlr
         * @return
         */
        public Builder minLearningRate(double mlr) {
            this.minLearningRate = mlr;
            return this;
        }

        /**
         * This method specifies number of iterations over batch on each node
         *
         * @param numIterations
         * @return
         */
        public Builder iterations(int numIterations) {
            this.numIterations = numIterations;
            return this;
        }

        /**
         * This method specifies number of epochs done over whole corpus
         *
         * PLEASE NOTE: NOT IMPLEMENTED
         *
         * @param numEpochs
         * @return
         */
        public Builder epochs(int numEpochs) {
            // TODO: implement epochs imitation for spark w2v
            this.numEpochs = numEpochs;
            return this;
        }

        /**
         * This method specifies minimum word frequency threshold. All words below this threshold will be ignored.
         *
         * @param minWordFrequency
         * @return
         */
        public Builder minWordFrequency(int minWordFrequency) {
            this.minWordFrequency = minWordFrequency;
            return this;
        }

        /**
         * This method specifies, if adaptive gradients should be used during model training
         *
         * @param reallyUse
         * @return
         */
        public Builder useAdaGrad(boolean reallyUse) {
            this.useAdaGrad = reallyUse;
            return this;
        }

        /**
         * Specifies random seed to be used during weights initialization;
         *
         * @param seed
         * @return
         */
        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        /**
         * Specifies TokenizerFactory to be used for tokenization
         *
         * PLEASE NOTE: You can't use anonymous implementation here
         *
         * @param factory
         * @return
         */
        public Builder tokenizerFactory(@NonNull TokenizerFactory factory) {
            this.tokenizer = factory.getClass().getCanonicalName();

            if (factory.getTokenPreProcessor() != null) {
                this.tokenPreprocessor = factory.getTokenPreProcessor().getClass().getCanonicalName();
            } else {
                this.tokenPreprocessor = "";
            }

            return this;
        }

        /**
         * Specifies TokenizerFactory class to be used for tokenization
         *
         *
         * @param tokenizer class name for tokenizerFactory
         * @return
         */
        public Builder tokenizerFactory(@NonNull String tokenizer) {
            this.tokenizer = tokenizer;
            return this;
        }

        /**
         * Specifies TokenPreProcessor class to be used during tokenization
         *
         *
         * @param tokenPreprocessor class name for tokenPreProcessor
         * @return
         */
        public Builder tokenPreprocessor(@NonNull String tokenPreprocessor) {
            this.tokenPreprocessor = tokenPreprocessor;
            return this;
        }

        /**
         * Specifies output vector's dimensions
         *
         * @param layerSize
         * @return
         */
        public Builder layerSize(int layerSize) {
            this.layerSize = layerSize;
            return this;
        }

        /**
         * Specifies N of n-Grams :)
         *
         * @param nGrams
         * @return
         */
        public Builder setNGrams(int nGrams) {
            this.nGrams = 1;
            return this;
        }

        /**
         * This method defines list of stop-words, that are to be ignored during vocab building and training
         *
         * @param stopWords
         * @return
         */
        public Builder stopWords(@NonNull List<String> stopWords) {
            for (String word: stopWords) {
                if (!this.stopWords.contains(word)) this.stopWords.add(word);
            }
            return this;
        }

        /**
         * Specifies the size of mini-batch, used in single iteration during training
         *
         * @param batchSize
         * @return
         */
        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        /**
         * Specifies, if UNK word should be used instead of words that are absent in vocab
         *
         * @param reallyUse
         * @return
         */
        public Builder useUnknown(boolean reallyUse) {
            this.useUnk = reallyUse;
            return this;
        }

        public Word2Vec build() {
            Word2Vec ret = new Word2Vec();

            this.configuration.setLearningRate(this.learningRate);
            this.configuration.setLayersSize(layerSize);
            this.configuration.setWindow(windowSize);
            this.configuration.setMinWordFrequency(minWordFrequency);
            this.configuration.setIterations(numIterations);
            this.configuration.setSeed(seed);
            this.configuration.setMinLearningRate(minLearningRate);
            this.configuration.setSampling(this.sampling);
            this.configuration.setUseAdaGrad(useAdaGrad);
            this.configuration.setNegative(negative);
            this.configuration.setEpochs(this.numEpochs);
            this.configuration.setBatchSize(this.batchSize);
            this.configuration.setStopList(this.stopWords);

            ret.configuration = this.configuration;

            ret.numEpochs = this.numEpochs;
            ret.numIterations = this.numIterations;
            ret.minWordFrequency = this.minWordFrequency;
            ret.learningRate.set(this.learningRate);
            ret.minLearningRate = this.minLearningRate;
            ret.sampling = this.sampling;
            ret.negative = this.negative;
            ret.layerSize = this.layerSize;
            ret.window = this.windowSize;
            ret.useAdeGrad = this.useAdaGrad;
            ret.stopWords = this.stopWords;
            ret.batchSize = this.batchSize;
            ret.useUnknown = this.useUnk;

            ret.tokenizer = this.tokenizer;
            ret.tokenPreprocessor = this.tokenPreprocessor;

            return ret;
        }
    }
}