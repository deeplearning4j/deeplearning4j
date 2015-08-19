package org.deeplearning4j.spark.text.functions;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.util.random.XORShiftRandom;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import scala.Tuple2;

import java.util.*;
import java.util.Map.Entry;

/**
 * @author jeffreytang
 */
public class FirstIterationFunction
        implements FlatMapFunction< Iterator<Tuple2<List<VocabWord>, Long>>, Entry<Integer, List<INDArray>> > {

    private int ithIteration;
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
    }

    @Override
    public Iterable<Entry<Integer, List<INDArray>>> call(Iterator<Tuple2<List<VocabWord>, Long>> pairIter) {

        Map<Integer, INDArray> indexSyn0VecMap = new HashMap<>();
        Map<Integer, INDArray> pointSyn1VecMap = new HashMap<>();
        Map<Integer, List<Integer>> pointIndexListMap = new HashMap<>();

        while (pairIter.hasNext()) {
            Tuple2<List<VocabWord>, Long> pair = pairIter.next();
            List<VocabWord> vocabWordsList = pair._1();
            Long sentenceCumSumCount = pair._2();
            double currentSentenceAlpha = Math.max(minAlpha,
                                          alpha - (alpha - minAlpha) * (sentenceCumSumCount / (double) totalWordCount));
            trainSentence(vocabWordsList, currentSentenceAlpha, indexSyn0VecMap, pointSyn1VecMap, pointIndexListMap);
        }
        return groupMap(indexSyn0VecMap, pointIndexListMap).entrySet();
    }


    public void trainSentence(List<VocabWord> vocabWordsList, double currentSentenceAlpha,
                              Map<Integer, INDArray> indexSyn0VecMap, Map<Integer, INDArray> pointSyn1VecMap,
                              Map<Integer, List<Integer>> pointIndexListMap) {

        if (vocabWordsList != null && !vocabWordsList.isEmpty()) {
            for (int ithWordInSentence = 0; ithWordInSentence < vocabWordsList.size(); ithWordInSentence++) {
                // Random value ranging from 0 to window size
                XORShiftRandom rand = new XORShiftRandom(seed ^ ((1 + ithWordInSentence) << 16) ^ ((-2 - ithIteration) << 8));
                int b = rand.nextInt(window);
                VocabWord currentWord = vocabWordsList.get(ithWordInSentence);
                if (currentWord != null) {
                    skipGram(ithWordInSentence, vocabWordsList, b,
                            currentSentenceAlpha, indexSyn0VecMap, pointSyn1VecMap, pointIndexListMap);
                }
            }
        }
    }

    public void skipGram(int ithWordInSentence, List<VocabWord> vocabWordsList, int b, double currentSentenceAlpha,
                         Map<Integer, INDArray> indexSyn0VecMap,  Map<Integer, INDArray> pointSyn1VecMap,
                         Map<Integer, List<Integer>> pointIndexListMap) {

        VocabWord currentWord = vocabWordsList.get(ithWordInSentence);
        if (currentWord != null && !vocabWordsList.isEmpty()) {
            int end = window * 2 + 1 - b;
            for (int a = b; a < end; a++) {
                if (a != window) {
                    int c = ithWordInSentence - window + a;
                    if (c >= 0 && c < vocabWordsList.size()) {
                        VocabWord lastWord = vocabWordsList.get(c);
                        iterateSample(currentWord, lastWord, currentSentenceAlpha,
                                      indexSyn0VecMap, pointSyn1VecMap, pointIndexListMap);
                    }
                }
            }
        }
    }

    public void iterateSample(VocabWord currentWord, VocabWord w2, double currentSentenceAlpha,
                              Map<Integer, INDArray> indexSyn0VecMap,  Map<Integer, INDArray> pointSyn1VecMap,
                              Map<Integer, List<Integer>> pointIndexListMap) {

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

            // Point to index
            if (pointIndexListMap.containsKey(point)) {
                pointIndexListMap.get(point).add(currentWordIndex);
            } else {
                pointIndexListMap.put(point, new ArrayList<Integer>() {{ add(currentWordIndex); }});
            }

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
            double g = (1 - code - f) * (useAdaGrad ? currentWord.getGradient(i, currentSentenceAlpha) : currentSentenceAlpha);


            Nd4j.getBlasWrapper().level1().axpy(vectorLength, g, syn1VecCurrentIndex, neu1e);
            Nd4j.getBlasWrapper().level1().axpy(vectorLength, g, randomSyn0Vec, syn1VecCurrentIndex);
        }

        // Updated the Syn0 vector based on gradient. Syn0 is not random anymore.
        Nd4j.getBlasWrapper().level1().axpy(vectorLength, 1.0f, neu1e, randomSyn0Vec);

        indexSyn0VecMap.put(currentWordIndex, randomSyn0Vec);
    }

    public Map<Integer, List<INDArray>> groupMap(Map<Integer, INDArray> indexSyn0VecMap,
                                           Map<Integer, List<Integer>> pointIndexListMap) {

        Map<Integer, List<INDArray>> pointSyn0VecMap = new HashMap<>();
        for (Entry<Integer, List<Integer>> pointIndexList : pointIndexListMap.entrySet()) {
            int point = pointIndexList.getKey();
            List<Integer> indexList = pointIndexList.getValue();
            for (int i=0; i < indexList.size(); i++) {
                int index = indexList.get(i);
                INDArray syn0Vec = indexSyn0VecMap.get(index);
                if (!pointSyn0VecMap.containsKey(point)) {
                    ArrayList<INDArray> indArrays = new ArrayList<>();
                    pointSyn0VecMap.put(point, indArrays);
                    indArrays.add(syn0Vec);
                } else {
                    pointSyn0VecMap.get(point).add(syn0Vec);
                }
            }
        }
        return pointSyn0VecMap;
    }

    public INDArray getRandomSyn0Vec(int vectorLength) {
        return Nd4j.rand(seed, new int[]{1 ,vectorLength}).subi(0.5).divi(vectorLength);
    }


}
