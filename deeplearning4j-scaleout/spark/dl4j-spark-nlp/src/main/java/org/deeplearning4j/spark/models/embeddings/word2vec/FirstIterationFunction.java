package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
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
        this.expTable = expTableBroadcast.getValue();
        this.vectorLength = (int) word2vecVarMap.get("vectorLength");
        this.useAdaGrad = (boolean) word2vecVarMap.get("useAdaGrad");
        this.negative = (int) word2vecVarMap.get("negative");
        this.window = (int) word2vecVarMap.get("window");
        this.alpha = (double) word2vecVarMap.get("alpha");
        this.minAlpha = (double) word2vecVarMap.get("minAlpha");
        this.totalWordCount = (long) word2vecVarMap.get("totalWordCount");
        this.seed = (long) word2vecVarMap.get("seed");
        this.maxExp = (int) word2vecVarMap.get("maxExp");
        this.indexSyn0VecMap = new HashMap<>();
        this.pointSyn1VecMap = new HashMap<>();
    }

    @Override
    public Iterable<Entry<Integer, INDArray>> call(Iterator<Tuple2<List<VocabWord>, Long>> pairIter) {

        while (pairIter.hasNext()) {
            Tuple2<List<VocabWord>, Long> pair = pairIter.next();
            List<VocabWord> vocabWordsList = pair._1();
            Long sentenceCumSumCount = pair._2();
//            System.out.println("Training sentence: " + vocabWordsList);
            double currentSentenceAlpha = Math.max(minAlpha,
                                          alpha - (alpha - minAlpha) * (sentenceCumSumCount / (double) totalWordCount));
            trainSentence(vocabWordsList, currentSentenceAlpha);
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

    public void iterateSample(VocabWord currentWord, VocabWord w2, double currentSentenceAlpha) {

        final int currentWordIndex = currentWord.getIndex();
        if (w2 == null || w2.getIndex() < 0 || currentWordIndex == w2.getIndex())
            return;

        // error for current word and context
        INDArray neu1e = Nd4j.create(vectorLength);

        // First iteration Syn0 is random numbers
        INDArray randomSyn0Vec = getRandomSyn0Vec(vectorLength);

        //
        for (int i = 0; i < currentWord.getCodeLength(); i++) {
            int code = currentWord.getCodes().get(i);
            int point = currentWord.getPoints().get(i);

            // Point to
            INDArray syn1VecCurrentIndex;
            if (pointSyn1VecMap.containsKey(point)) {
                syn1VecCurrentIndex = pointSyn1VecMap.get(point);
            } else {
                syn1VecCurrentIndex = Nd4j.zeros(1, vectorLength); // 1 row of vector length of zeros
                pointSyn1VecMap.put(point, syn1VecCurrentIndex);
            }

            // Dot product of Syn0 and Syn1 vecs
            double dot = Nd4j.getBlasWrapper().level1().dot(vectorLength, 1.0, randomSyn0Vec, syn1VecCurrentIndex);

            if (dot < -maxExp || dot >= maxExp)
                continue;

            int idx = (int) ((dot + maxExp) * ((double) expTable.length / maxExp / 2.0));

            //score
            double f = expTable[idx];
            //gradient
            double g = (1 - code - f) * (useAdaGrad ? currentWord.getGradient(i, currentSentenceAlpha, currentSentenceAlpha) : currentSentenceAlpha);


            Nd4j.getBlasWrapper().level1().axpy(vectorLength, g, syn1VecCurrentIndex, neu1e);
            Nd4j.getBlasWrapper().level1().axpy(vectorLength, g, randomSyn0Vec, syn1VecCurrentIndex);
        }

        // Updated the Syn0 vector based on gradient. Syn0 is not random anymore.
        Nd4j.getBlasWrapper().level1().axpy(vectorLength, 1.0f, neu1e, randomSyn0Vec);

        indexSyn0VecMap.put(currentWordIndex, randomSyn0Vec);
    }

    public INDArray getRandomSyn0Vec(int vectorLength) {
        return Nd4j.rand(seed, new int[]{1 ,vectorLength}).subi(0.5).divi(vectorLength);
    }
}
