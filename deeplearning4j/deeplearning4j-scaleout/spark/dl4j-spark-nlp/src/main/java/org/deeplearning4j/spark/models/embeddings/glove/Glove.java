/*-
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

package org.deeplearning4j.spark.models.embeddings.glove;

import org.apache.commons.math3.util.FastMath;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.glove.GloveWeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.models.embeddings.glove.cooccurrences.CoOccurrenceCalculator;
import org.deeplearning4j.spark.models.embeddings.glove.cooccurrences.CoOccurrenceCounts;
import org.deeplearning4j.spark.text.functions.TextPipeline;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.legacy.AdaGrad;
import org.nd4j.linalg.primitives.CounterMap;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

import static org.deeplearning4j.spark.models.embeddings.word2vec.Word2VecVariables.*;

/**
 * Spark glove
 *
 * @author Adam Gibson
 */
public class Glove implements Serializable {

    private Broadcast<VocabCache<VocabWord>> vocabCacheBroadcast;
    private String tokenizerFactoryClazz = DefaultTokenizerFactory.class.getName();
    private boolean symmetric = true;
    private int windowSize = 15;
    private int iterations = 300;
    private static Logger log = LoggerFactory.getLogger(Glove.class);

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


    private Pair<INDArray, Float> update(AdaGrad weightAdaGrad, AdaGrad biasAdaGrad, INDArray syn0, INDArray bias,
                    VocabWord w1, INDArray wordVector, INDArray contextVector, double gradient) {
        //gradient for word vectors
        INDArray grad1 = contextVector.mul(gradient);
        INDArray update = weightAdaGrad.getGradient(grad1, w1.getIndex(), ArrayUtil.toInts(syn0.shape()));
        wordVector.subi(update);

        double w1Bias = bias.getDouble(w1.getIndex());
        double biasGradient = biasAdaGrad.getGradient(gradient, w1.getIndex(), ArrayUtil.toInts(bias.shape()));
        double update2 = w1Bias - biasGradient;
        bias.putScalar(w1.getIndex(), bias.getDouble(w1.getIndex()) - update2);
        return new Pair<>(update, (float) update2);
    }

    /**
     * Train on the corpus
     * @param rdd the rdd to train
     * @return the vocab and weights
     */
    public Pair<VocabCache<VocabWord>, GloveWeightLookupTable> train(JavaRDD<String> rdd) throws Exception {
        // Each `train()` can use different parameters
        final JavaSparkContext sc = new JavaSparkContext(rdd.context());
        final SparkConf conf = sc.getConf();
        final int vectorLength = assignVar(VECTOR_LENGTH, conf, Integer.class);
        final boolean useAdaGrad = assignVar(ADAGRAD, conf, Boolean.class);
        final double negative = assignVar(NEGATIVE, conf, Double.class);
        final int numWords = assignVar(NUM_WORDS, conf, Integer.class);
        final int window = assignVar(WINDOW, conf, Integer.class);
        final double alpha = assignVar(ALPHA, conf, Double.class);
        final double minAlpha = assignVar(MIN_ALPHA, conf, Double.class);
        final int iterations = assignVar(ITERATIONS, conf, Integer.class);
        final int nGrams = assignVar(N_GRAMS, conf, Integer.class);
        final String tokenizer = assignVar(TOKENIZER, conf, String.class);
        final String tokenPreprocessor = assignVar(TOKEN_PREPROCESSOR, conf, String.class);
        final boolean removeStop = assignVar(REMOVE_STOPWORDS, conf, Boolean.class);

        Map<String, Object> tokenizerVarMap = new HashMap<String, Object>() {
            {
                put("numWords", numWords);
                put("nGrams", nGrams);
                put("tokenizer", tokenizer);
                put("tokenPreprocessor", tokenPreprocessor);
                put("removeStop", removeStop);
            }
        };
        Broadcast<Map<String, Object>> broadcastTokenizerVarMap = sc.broadcast(tokenizerVarMap);


        TextPipeline pipeline = new TextPipeline(rdd, broadcastTokenizerVarMap);
        pipeline.buildVocabCache();
        pipeline.buildVocabWordListRDD();


        // Get total word count
        Long totalWordCount = pipeline.getTotalWordCount();
        VocabCache<VocabWord> vocabCache = pipeline.getVocabCache();
        JavaRDD<Pair<List<String>, AtomicLong>> sentenceWordsCountRDD = pipeline.getSentenceWordsCountRDD();
        final Pair<VocabCache<VocabWord>, Long> vocabAndNumWords = new Pair<>(vocabCache, totalWordCount);

        vocabCacheBroadcast = sc.broadcast(vocabAndNumWords.getFirst());

        final GloveWeightLookupTable gloveWeightLookupTable = new GloveWeightLookupTable.Builder()
                        .cache(vocabAndNumWords.getFirst()).lr(conf.getDouble(GlovePerformer.ALPHA, 0.01))
                        .maxCount(conf.getDouble(GlovePerformer.MAX_COUNT, 100))
                        .vectorLength(conf.getInt(GlovePerformer.VECTOR_LENGTH, 300))
                        .xMax(conf.getDouble(GlovePerformer.X_MAX, 0.75)).build();
        gloveWeightLookupTable.resetWeights();

        gloveWeightLookupTable.getBiasAdaGrad().historicalGradient = Nd4j.ones(gloveWeightLookupTable.getSyn0().rows());
        gloveWeightLookupTable.getWeightAdaGrad().historicalGradient =
                        Nd4j.ones(gloveWeightLookupTable.getSyn0().shape());


        log.info("Created lookup table of size " + Arrays.toString(gloveWeightLookupTable.getSyn0().shape()));
        CounterMap<String, String> coOccurrenceCounts = sentenceWordsCountRDD
                        .map(new CoOccurrenceCalculator(symmetric, vocabCacheBroadcast, windowSize))
                        .fold(new CounterMap<String, String>(), new CoOccurrenceCounts());
        Iterator<Pair<String, String>> pair2 = coOccurrenceCounts.getIterator();
        List<Triple<String, String, Float>> counts = new ArrayList<>();

        while (pair2.hasNext()) {
            Pair<String, String> next = pair2.next();
            if (coOccurrenceCounts.getCount(next.getFirst(), next.getSecond()) > gloveWeightLookupTable.getMaxCount()) {
                coOccurrenceCounts.setCount(next.getFirst(), next.getSecond(),
                                (float) gloveWeightLookupTable.getMaxCount());
            }
            counts.add(new Triple<>(next.getFirst(), next.getSecond(),
                    (float) coOccurrenceCounts.getCount(next.getFirst(), next.getSecond())));

        }

        log.info("Calculated co occurrences");

        JavaRDD<Triple<String, String, Float>> parallel = sc.parallelize(counts);
        JavaPairRDD<String, Tuple2<String, Float>> pairs = parallel
                        .mapToPair(new PairFunction<Triple<String, String, Float>, String, Tuple2<String, Float>>() {
                            @Override
                            public Tuple2<String, Tuple2<String, Float>> call(
                                            Triple<String, String, Float> stringStringDoubleTriple) throws Exception {
                                return new Tuple2<>(stringStringDoubleTriple.getFirst(),
                                                new Tuple2<>(stringStringDoubleTriple.getSecond(),
                                                                stringStringDoubleTriple.getThird()));
                            }
                        });

        JavaPairRDD<VocabWord, Tuple2<VocabWord, Float>> pairsVocab = pairs.mapToPair(
                        new PairFunction<Tuple2<String, Tuple2<String, Float>>, VocabWord, Tuple2<VocabWord, Float>>() {
                            @Override
                            public Tuple2<VocabWord, Tuple2<VocabWord, Float>> call(
                                            Tuple2<String, Tuple2<String, Float>> stringTuple2Tuple2) throws Exception {
                                VocabWord w1 = vocabCacheBroadcast.getValue().wordFor(stringTuple2Tuple2._1());
                                VocabWord w2 = vocabCacheBroadcast.getValue().wordFor(stringTuple2Tuple2._2()._1());
                                return new Tuple2<>(w1, new Tuple2<>(w2, stringTuple2Tuple2._2()._2()));
                            }
                        });


        for (int i = 0; i < iterations; i++) {
            JavaRDD<GloveChange> change =
                            pairsVocab.map(new Function<Tuple2<VocabWord, Tuple2<VocabWord, Float>>, GloveChange>() {
                                @Override
                                public GloveChange call(
                                                Tuple2<VocabWord, Tuple2<VocabWord, Float>> vocabWordTuple2Tuple2)
                                                throws Exception {
                                    VocabWord w1 = vocabWordTuple2Tuple2._1();
                                    VocabWord w2 = vocabWordTuple2Tuple2._2()._1();
                                    INDArray w1Vector = gloveWeightLookupTable.getSyn0().slice(w1.getIndex());
                                    INDArray w2Vector = gloveWeightLookupTable.getSyn0().slice(w2.getIndex());
                                    INDArray bias = gloveWeightLookupTable.getBias();
                                    double score = vocabWordTuple2Tuple2._2()._2();
                                    double xMax = gloveWeightLookupTable.getxMax();
                                    double maxCount = gloveWeightLookupTable.getMaxCount();
                                    //w1 * w2 + bias
                                    double prediction = Nd4j.getBlasWrapper().dot(w1Vector, w2Vector);
                                    prediction += bias.getDouble(w1.getIndex()) + bias.getDouble(w2.getIndex());

                                    double weight = FastMath.pow(Math.min(1.0, (score / maxCount)), xMax);

                                    double fDiff = score > xMax ? prediction : weight * (prediction - Math.log(score));
                                    if (Double.isNaN(fDiff))
                                        fDiff = Nd4j.EPS_THRESHOLD;
                                    //amount of change
                                    double gradient = fDiff;

                                    Pair<INDArray, Float> w1Update = update(gloveWeightLookupTable.getWeightAdaGrad(),
                                                    gloveWeightLookupTable.getBiasAdaGrad(),
                                                    gloveWeightLookupTable.getSyn0(), gloveWeightLookupTable.getBias(),
                                                    w1, w1Vector, w2Vector, gradient);
                                    Pair<INDArray, Float> w2Update = update(gloveWeightLookupTable.getWeightAdaGrad(),
                                                    gloveWeightLookupTable.getBiasAdaGrad(),
                                                    gloveWeightLookupTable.getSyn0(), gloveWeightLookupTable.getBias(),
                                                    w2, w2Vector, w1Vector, gradient);
                                    return new GloveChange(w1, w2, w1Update.getFirst(), w2Update.getFirst(),
                                                    w1Update.getSecond(), w2Update.getSecond(), fDiff,
                                                    gloveWeightLookupTable.getWeightAdaGrad().getHistoricalGradient()
                                                                    .slice(w1.getIndex()),
                                                    gloveWeightLookupTable.getWeightAdaGrad().getHistoricalGradient()
                                                                    .slice(w2.getIndex()),
                                                    gloveWeightLookupTable.getBiasAdaGrad().getHistoricalGradient()
                                                                    .getDouble(w2.getIndex()),
                                                    gloveWeightLookupTable.getBiasAdaGrad().getHistoricalGradient()
                                                                    .getDouble(w1.getIndex()));

                                }
                            });



            List<GloveChange> gloveChanges = change.collect();
            double error = 0.0;
            for (GloveChange change2 : gloveChanges) {
                change2.apply(gloveWeightLookupTable);
                error += change2.getError();
            }


            List l = pairsVocab.collect();
            Collections.shuffle(l);
            pairsVocab = sc.parallelizePairs(l);

            log.info("Error at iteration " + i + " was " + error);



        }

        return new Pair<>(vocabAndNumWords.getFirst(), gloveWeightLookupTable);
    }

}
