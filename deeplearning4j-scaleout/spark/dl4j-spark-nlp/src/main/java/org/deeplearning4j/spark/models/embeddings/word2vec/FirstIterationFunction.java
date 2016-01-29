package org.deeplearning4j.spark.models.embeddings.word2vec;

import lombok.NonNull;
import org.apache.commons.math3.util.FastMath;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author jeffreytang
 */
public class FirstIterationFunction
        implements FlatMapFunction< Iterator<Tuple2<List<VocabWord>, Long>>, Entry<Integer, INDArray> > {

    private int ithIteration = 1;
    private int vectorLength;
    private boolean useAdaGrad;
    private int negative;
    private int window;
    private double alpha;
    private double minAlpha;
    private long totalWordCount;
    private long seed;
    private int maxExp;
    private double[] expTable;
    private Map<Integer, INDArray> indexSyn0VecMap;
    private Map<Integer, INDArray> pointSyn1VecMap;
    private AtomicLong nextRandom = new AtomicLong(5);


    public FirstIterationFunction(Broadcast<Map<String, Object>> word2vecVarMapBroadcast,
                                  Broadcast<double[]> expTableBroadcast) {

        Map<String, Object> word2vecVarMap = word2vecVarMapBroadcast.getValue();
       // this.expTable = expTableBroadcast.getValue();
        this.vectorLength = (int) word2vecVarMap.get("vectorLength");
        this.useAdaGrad = (boolean) word2vecVarMap.get("useAdaGrad");
        this.negative = (int) word2vecVarMap.get("negative");
        this.window = (int) word2vecVarMap.get("window");
        this.alpha = (double) word2vecVarMap.get("alpha");
        this.minAlpha = (double) word2vecVarMap.get("minAlpha");
        this.totalWordCount = (long) word2vecVarMap.get("totalWordCount");
        this.seed = (long) word2vecVarMap.get("seed");
        this.maxExp = (int) 6; // word2vecVarMap.get("maxExp");
        this.indexSyn0VecMap = new HashMap<>();
        this.pointSyn1VecMap = new HashMap<>();

        this.expTable = new double[100000];
        for (int i = 0; i < expTable.length; i++) {
            double tmp =   FastMath.exp((i / (double) expTable.length * 2 - 1) * maxExp);
            expTable[i]  = tmp / (tmp + 1.0);
        }
    }

    @Override
    public Iterable<Entry<Integer, INDArray>> call(Iterator<Tuple2<List<VocabWord>, Long>> pairIter) {
        int cnt = 0;
        while (pairIter.hasNext()) {
            Tuple2<List<VocabWord>, Long> pair = pairIter.next();
            List<VocabWord> vocabWordsList = pair._1();
            Long sentenceCumSumCount = pair._2();
            //System.out.println("Training sentence: " + vocabWordsList);
            double currentSentenceAlpha = Math.max(minAlpha,
                                          alpha - (alpha - minAlpha) * (sentenceCumSumCount / (double) totalWordCount));
            trainSentence(vocabWordsList, currentSentenceAlpha);
            cnt++;
        }
        System.out.println("Blocked calls: " + cnt);
        System.out.println("two/four internal: " + Transforms.cosineSim(indexSyn0VecMap.get(126),indexSyn0VecMap.get(173)));
        System.out.println("Two internal: " + indexSyn0VecMap.get(126));
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

        // Updated the Syn0 vector based on gradient. Syn0 is not random anymore.
        Nd4j.getBlasWrapper().level1().axpy(vectorLength, 1.0f, neu1e, l1);

        indexSyn0VecMap.put(currentWordIndex, l1);
    }

    public INDArray getRandomSyn0Vec(int vectorLength, long lseed) {
        return Nd4j.rand(lseed, new int[]{1 ,vectorLength}).subi(0.5).divi(vectorLength);
    }
}
