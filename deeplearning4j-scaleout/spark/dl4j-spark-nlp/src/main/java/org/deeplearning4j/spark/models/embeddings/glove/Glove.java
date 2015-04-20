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

package org.deeplearning4j.spark.models.embeddings.glove;

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
import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.models.glove.GloveWeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.models.embeddings.glove.cooccurrences.CoOccurrenceCalculator;
import org.deeplearning4j.spark.models.embeddings.glove.cooccurrences.CoOccurrenceCounts;
import org.deeplearning4j.spark.text.TextPipeline;
import org.deeplearning4j.spark.text.TokenizerFunction;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdaGrad;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.io.Serializable;
import java.util.*;

/**
 * Spark glove
 *
 * @author Adam Gibson
 */
public class Glove implements Serializable {

    private Broadcast<VocabCache> vocabCacheBroadcast;
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


    private Pair<INDArray,Double> update(AdaGrad weightAdaGrad,AdaGrad biasAdaGrad,INDArray syn0,INDArray bias,VocabWord w1,INDArray wordVector,INDArray contextVector,double gradient) {
        //gradient for word vectors
        INDArray grad1 =  contextVector.mul(gradient);
        INDArray update = weightAdaGrad.getGradient(grad1,w1.getIndex(),syn0.shape());


        double w1Bias = bias.getDouble(w1.getIndex());
        double biasGradient = biasAdaGrad.getGradient(gradient,w1.getIndex(),bias.shape());
        double update2 = w1Bias - biasGradient;
        return new Pair<>(update,update2);
    }

    /**
     * Train on the corpus
     * @param rdd the rdd to train
     * @return the vocab and weights
     */
    public Pair<VocabCache,GloveWeightLookupTable> train(JavaRDD<String> rdd) {
        TextPipeline pipeline = new TextPipeline(rdd);
        final Pair<VocabCache,Long> vocabAndNumWords = pipeline.process();
        SparkConf conf = rdd.context().getConf();
        JavaSparkContext sc = new JavaSparkContext(rdd.context());
        vocabCacheBroadcast = sc.broadcast(vocabAndNumWords.getFirst());

        final GloveWeightLookupTable gloveWeightLookupTable = new GloveWeightLookupTable.Builder()
                .cache(vocabAndNumWords.getFirst()).lr(conf.getDouble(GlovePerformer.ALPHA,0.025))
                .maxCount(conf.getDouble(GlovePerformer.MAX_COUNT,100)).vectorLength(conf.getInt(GlovePerformer.VECTOR_LENGTH,300))
                .xMax(conf.getDouble(GlovePerformer.X_MAX,0.75)).build();
        gloveWeightLookupTable.resetWeights();

        gloveWeightLookupTable.getBiasAdaGrad().historicalGradient = Nd4j.zeros(gloveWeightLookupTable.getSyn0().rows());
        gloveWeightLookupTable.getWeightAdaGrad().historicalGradient = Nd4j.create(gloveWeightLookupTable.getSyn0().shape());



        log.info("Created lookup table of size " + Arrays.toString(gloveWeightLookupTable.getSyn0().shape()));
        CounterMap<String,String> coOccurrenceCounts = rdd.map(new TokenizerFunction(tokenizerFactoryClazz))
                .map(new CoOccurrenceCalculator(symmetric,vocabCacheBroadcast,windowSize)).fold(new CounterMap<String, String>(),new CoOccurrenceCounts());

        List<Triple<String,String,Double>> counts = new ArrayList<>();
        Iterator<Pair<String,String>> pairIter = coOccurrenceCounts.getPairIterator();
        while(pairIter.hasNext()) {
            Pair<String,String> pair = pairIter.next();
            counts.add(new Triple<>(pair.getFirst(),pair.getSecond(),coOccurrenceCounts.getCount(pair.getFirst(),pair.getSecond())));
        }

        log.info("Calculated co occurrences");

        JavaRDD<Triple<String,String,Double>> parallel = sc.parallelize(counts);
        JavaPairRDD<String, Tuple2<String,Double>> pairs = parallel.mapToPair(new PairFunction<Triple<String, String, Double>, String, Tuple2<String, Double>>() {
            @Override
            public Tuple2<String, Tuple2<String, Double>> call(Triple<String, String, Double> stringStringDoubleTriple) throws Exception {
                return new Tuple2<>(stringStringDoubleTriple.getFirst(),new Tuple2<>(stringStringDoubleTriple.getFirst(),stringStringDoubleTriple.getThird()));
            }
        });

        JavaPairRDD<VocabWord,Tuple2<VocabWord,Double>> pairsVocab = pairs.mapToPair(new PairFunction<Tuple2<String, Tuple2<String, Double>>, VocabWord, Tuple2<VocabWord, Double>>() {
            @Override
            public Tuple2<VocabWord, Tuple2<VocabWord, Double>> call(Tuple2<String, Tuple2<String, Double>> stringTuple2Tuple2) throws Exception {
                return new Tuple2<>(vocabCacheBroadcast.getValue().wordFor(stringTuple2Tuple2._1())
                        , new Tuple2<>(vocabCacheBroadcast.getValue().wordFor(stringTuple2Tuple2._2()._1()),stringTuple2Tuple2._2()._2()));
            }
        });


        for(int i = 0; i < iterations; i++) {

            JavaRDD<GloveChange> change = pairsVocab.map(new Function<Tuple2<VocabWord,Tuple2<VocabWord,Double>>, GloveChange>() {
                @Override
                public GloveChange call(Tuple2<VocabWord, Tuple2<VocabWord, Double>> vocabWordTuple2Tuple2) throws Exception {
                    VocabWord w1 = vocabWordTuple2Tuple2._1();
                    VocabWord w2 = vocabWordTuple2Tuple2._2()._1();
                    INDArray w1Vector =  gloveWeightLookupTable.getSyn0().slice(w1.getIndex());
                    INDArray w2Vector = gloveWeightLookupTable.getSyn0().slice(w2.getIndex());
                    INDArray bias = gloveWeightLookupTable.getBias();
                    double score = vocabWordTuple2Tuple2._2()._2();
                    double xMax = gloveWeightLookupTable.getxMax();
                    double maxCount = gloveWeightLookupTable.getMaxCount();
                    //w1 * w2 + bias
                    double prediction = Nd4j.getBlasWrapper().dot(w1Vector,w2Vector);
                    prediction +=  bias.getDouble(w1.getIndex()) + bias.getDouble(w2.getIndex());

                    double weight = Math.pow(Math.min(1.0,(score / maxCount)),xMax);

                    double fDiff = score > xMax ? prediction :  weight * (prediction - Math.log(score));
                    if(Double.isNaN(fDiff))
                        fDiff = Nd4j.EPS_THRESHOLD;
                    //amount of change
                    double gradient =  fDiff;
                    // update(w1,w1Vector,w2Vector,gradient);
                    //update(w2,w2Vector,w1Vector,gradient);

                    Pair<INDArray,Double> w1Update = update(
                            gloveWeightLookupTable.getWeightAdaGrad()
                            ,gloveWeightLookupTable.getBiasAdaGrad()
                            ,gloveWeightLookupTable.getSyn0()
                            ,gloveWeightLookupTable.getBias(),w1,w1Vector,w2Vector,gradient);
                    Pair<INDArray,Double> w2Update =  update(
                            gloveWeightLookupTable.getWeightAdaGrad()
                            ,gloveWeightLookupTable.getBiasAdaGrad()
                            ,gloveWeightLookupTable.getSyn0()
                            ,gloveWeightLookupTable.getBias(),w2,w2Vector,w1Vector,gradient);
                    return new GloveChange(w1,w2,w1Update.getFirst(),w2Update.getFirst(),w1Update.getSecond(),w2Update.getSecond(),fDiff);
                }
            });

            JavaRDD<Double> error = change.map(new Function<GloveChange,Double>() {
                @Override
                public Double call(GloveChange gloveChange) throws Exception {
                    gloveChange.apply(gloveWeightLookupTable);
                    return gloveChange.getError();
                }
            });

            final Accumulator<Double> d = sc.accumulator(0.0);
            error.foreach(new VoidFunction<Double>() {
                @Override
                public void call(Double aDouble) throws Exception {
                    d.$plus$eq(aDouble);
                }
            });

            log.info("Error at iteration " + i + " was " + d.value());



        }

        return new Pair<>(vocabAndNumWords.getFirst(),gloveWeightLookupTable);
    }

}
