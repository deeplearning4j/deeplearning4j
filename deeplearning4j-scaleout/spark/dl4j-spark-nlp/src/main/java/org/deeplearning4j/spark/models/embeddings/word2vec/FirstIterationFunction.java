package org.deeplearning4j.spark.models.embeddings.word2vec;

import lombok.NonNull;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.util.FastMath;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import scala.Tuple2;

import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author jeffreytang
 * @author raver119@gmail.com
 */
public class FirstIterationFunction
        implements FlatMapFunction< Iterator<Tuple2<List<VocabWord>, Long>>, Entry<Integer, INDArray> > {

    private int ithIteration = 1;
    private int vectorLength;
    private boolean useAdaGrad;
    private int batchSize = 0;
    private double negative;
    private int window;
    private double alpha;
    private double minAlpha;
    private long totalWordCount;
    private long seed;
    private int maxExp;
    private double[] expTable;
    private int iterations;
    private Map<Integer, INDArray> indexSyn0VecMap;
    private Map<Integer, INDArray> pointSyn1VecMap;
    private AtomicLong nextRandom = new AtomicLong(5);

    private volatile VocabCache<VocabWord> vocab;
    private volatile NegativeHolder negativeHolder;




    public FirstIterationFunction(Broadcast<Map<String, Object>> word2vecVarMapBroadcast,
                                  Broadcast<double[]> expTableBroadcast, Broadcast<VocabCache<VocabWord>> vocabCacheBroadcast) {

        Map<String, Object> word2vecVarMap = word2vecVarMapBroadcast.getValue();
        this.expTable = expTableBroadcast.getValue();
        this.vectorLength = (int) word2vecVarMap.get("vectorLength");
        this.useAdaGrad = (boolean) word2vecVarMap.get("useAdaGrad");
        this.negative = (double) word2vecVarMap.get("negative");
        this.window = (int) word2vecVarMap.get("window");
        this.alpha = (double) word2vecVarMap.get("alpha");
        this.minAlpha = (double) word2vecVarMap.get("minAlpha");
        this.totalWordCount = (long) word2vecVarMap.get("totalWordCount");
        this.seed = (long) word2vecVarMap.get("seed");
        this.maxExp = (int) word2vecVarMap.get("maxExp");
        this.iterations = (int) word2vecVarMap.get("iterations");
        this.batchSize = (int) word2vecVarMap.get("batchSize");
        this.indexSyn0VecMap = new HashMap<>();
        this.pointSyn1VecMap = new HashMap<>();
        this.vocab = vocabCacheBroadcast.getValue();

        if (this.vocab == null) throw new RuntimeException("VocabCache is null");

        if (negative > 0) {
            negativeHolder = NegativeHolder.getInstance();
            negativeHolder.initHolder(vocab, expTable, this.vectorLength);
        }
    }



    @Override
    public Iterable<Entry<Integer, INDArray>> call(Iterator<Tuple2<List<VocabWord>, Long>> pairIter) {
        while (pairIter.hasNext()) {
            List<Pair<List<VocabWord>, Long>> batch = new ArrayList<>();
            while (pairIter.hasNext() && batch.size() < batchSize) {
                Tuple2<List<VocabWord>, Long> pair = pairIter.next();
                List<VocabWord> vocabWordsList = pair._1();
                Long sentenceCumSumCount = pair._2();
                batch.add(Pair.of(vocabWordsList, sentenceCumSumCount));
            }

            for (int i = 0; i < iterations; i++) {
                //System.out.println("Training sentence: " + vocabWordsList);
                for (Pair<List<VocabWord>, Long> pair: batch) {
                    List<VocabWord> vocabWordsList = pair.getKey();
                    Long sentenceCumSumCount = pair.getValue();
                    double currentSentenceAlpha = Math.max(minAlpha,
                            alpha - (alpha - minAlpha) * (sentenceCumSumCount / (double) totalWordCount));
                    trainSentence(vocabWordsList, currentSentenceAlpha);
                }
            }
        }
        return indexSyn0VecMap.entrySet();
    }


    public void trainSentence(List<VocabWord> vocabWordsList, double currentSentenceAlpha) {

        if (vocabWordsList != null && !vocabWordsList.isEmpty()) {
            for (int ithWordInSentence = 0; ithWordInSentence < vocabWordsList.size(); ithWordInSentence++) {
                // Random value ranging from 0 to window size
                nextRandom.set(nextRandom.get() * 25214903917L + 11);
                int b = (int) (long) this.nextRandom.get() % window;
                VocabWord currentWord = vocabWordsList.get(ithWordInSentence);
                if (currentWord != null) {
                    skipGram(ithWordInSentence, vocabWordsList, b, currentSentenceAlpha);
                }
            }
        }
    }

    public void skipGram(int ithWordInSentence, List<VocabWord> vocabWordsList, int b, double currentSentenceAlpha) {

        VocabWord currentWord = vocabWordsList.get(ithWordInSentence);
        if (currentWord != null && !vocabWordsList.isEmpty()) {
            int end = window * 2 + 1 - b;
            for (int a = b; a < end; a++) {
                if (a != window) {
                    int c = ithWordInSentence - window + a;
                    if (c >= 0 && c < vocabWordsList.size()) {
                        VocabWord lastWord = vocabWordsList.get(c);
                        iterateSample(currentWord, lastWord, currentSentenceAlpha);
                    }
                }
            }
        }
    }

    public void iterateSample(VocabWord w1, VocabWord w2, double currentSentenceAlpha) {


        if (w1 == null || w2 == null || w2.getIndex() < 0 || w2.getIndex() == w1.getIndex())
            return;
        final int currentWordIndex = w2.getIndex();

        // error for current word and context
        INDArray neu1e = Nd4j.create(vectorLength);

        // First iteration Syn0 is random numbers
        INDArray l1 = null;
        if (indexSyn0VecMap.containsKey(currentWordIndex)) {
            l1 = indexSyn0VecMap.get(currentWordIndex);
        } else {
            l1 = getRandomSyn0Vec(vectorLength, (long) currentWordIndex);
        }

        //
        for (int i = 0; i < w1.getCodeLength(); i++) {
            int code = w1.getCodes().get(i);
            int point = w1.getPoints().get(i);
            if(point < 0)
                throw new IllegalStateException("Illegal point " + point);
            // Point to
            INDArray syn1;
            if (pointSyn1VecMap.containsKey(point)) {
                syn1 = pointSyn1VecMap.get(point);
            } else {
                syn1 = Nd4j.zeros(1, vectorLength); // 1 row of vector length of zeros
                pointSyn1VecMap.put(point, syn1);
            }

            // Dot product of Syn0 and Syn1 vecs
            double dot = Nd4j.getBlasWrapper().level1().dot(vectorLength, 1.0, l1, syn1);

            if (dot < -maxExp || dot >= maxExp)
                continue;

            int idx = (int) ((dot + maxExp) * ((double) expTable.length / maxExp / 2.0));

            if (idx > expTable.length) continue;

            //score
            double f = expTable[idx];
            //gradient
            double g = (1 - code - f) * (useAdaGrad ? w1.getGradient(i, currentSentenceAlpha, currentSentenceAlpha) : currentSentenceAlpha);


            Nd4j.getBlasWrapper().level1().axpy(vectorLength, g, syn1, neu1e);
            Nd4j.getBlasWrapper().level1().axpy(vectorLength, g, l1, syn1);
        }

        int target = w1.getIndex();
        int label;
        //negative sampling
        if(negative > 0)
            for (int d = 0; d < negative + 1; d++) {
                if (d == 0)
                    label = 1;
                else {
                    nextRandom.set(nextRandom.get() * 25214903917L + 11);
                    int idx = Math.abs((int) (nextRandom.get() >> 16) % negativeHolder.getTable().length());

                    target = negativeHolder.getTable().getInt(idx);
                    if (target <= 0)
                        target = (int) nextRandom.get() % (vocab.numWords() - 1) + 1;

                    if (target == w1.getIndex())
                        continue;
                    label = 0;
                }

                if(target >= negativeHolder.getSyn1Neg().rows() || target < 0)
                    continue;

                double f = Nd4j.getBlasWrapper().dot(l1,negativeHolder.getSyn1Neg().slice(target));
                double g;
                if (f > maxExp)
                    g = useAdaGrad ? w1.getGradient(target, (label - 1), alpha) : (label - 1) *  alpha;
                else if (f < -maxExp)
                    g = label * (useAdaGrad ?  w1.getGradient(target, alpha, alpha) : alpha);
                else {
                    int idx = (int) ((f + maxExp) * (expTable.length / maxExp / 2));
                    if (idx >= expTable.length)
                        continue;

                    g = useAdaGrad ? w1.getGradient(target, label - expTable[idx], alpha) : (label - expTable[idx]) * alpha;
                }

                    Nd4j.getBlasWrapper().axpy((float) g,negativeHolder.getSyn1Neg().slice(target),neu1e);

                    Nd4j.getBlasWrapper().axpy((float) g,l1,negativeHolder.getSyn1Neg().slice(target));
            }


        // Updated the Syn0 vector based on gradient. Syn0 is not random anymore.
        Nd4j.getBlasWrapper().level1().axpy(vectorLength, 1.0f, neu1e, l1);

        indexSyn0VecMap.put(currentWordIndex, l1);
    }

    private INDArray getRandomSyn0Vec(int vectorLength, long lseed) {
        /*
            we use wordIndex as part of seed here, to guarantee that during word syn0 initialization on dwo distinct nodes, initial weights will be the same for the same word
         */
        return Nd4j.rand(lseed * seed, new int[]{1 ,vectorLength}).subi(0.5).divi(vectorLength);
    }
}
