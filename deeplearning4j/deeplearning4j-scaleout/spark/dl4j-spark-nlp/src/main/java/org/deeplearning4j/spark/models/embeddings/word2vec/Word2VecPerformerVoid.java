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

package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.commons.math3.util.FastMath;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Base line word 2 vec performer
 *
 * @author Adam Gibson
 */
@Deprecated
public class Word2VecPerformerVoid implements VoidFunction<Pair<List<VocabWord>, AtomicLong>> {


    public final static String NAME_SPACE = "org.deeplearning4j.scaleout.perform.models.word2vec";
    public final static String VECTOR_LENGTH = NAME_SPACE + ".length";
    public final static String ADAGRAD = NAME_SPACE + ".adagrad";
    public final static String NEGATIVE = NAME_SPACE + ".negative";
    public final static String NUM_WORDS = NAME_SPACE + ".numwords";
    public final static String TABLE = NAME_SPACE + ".table";
    public final static String WINDOW = NAME_SPACE + ".window";
    public final static String ALPHA = NAME_SPACE + ".alpha";
    public final static String MIN_ALPHA = NAME_SPACE + ".minalpha";
    public final static String ITERATIONS = NAME_SPACE + ".iterations";

    private static double MAX_EXP = 6;
    private boolean useAdaGrad = false;
    private double negative = 5;
    private int numWords = 1;
    private INDArray table;
    private int window = 5;
    private AtomicLong nextRandom = new AtomicLong(5);
    private double alpha = 0.025;
    private double minAlpha = 1e-2;
    private int totalWords = 1;
    private int iterations = 5;
    private static transient final Logger log = LoggerFactory.getLogger(Word2VecPerformerVoid.class);
    private int lastChecked = 0;
    private Broadcast<AtomicLong> wordCount;
    private InMemoryLookupTable weights;
    private double[] expTable = new double[1000];
    private int vectorLength;


    public Word2VecPerformerVoid(SparkConf sc, Broadcast<AtomicLong> wordCount, InMemoryLookupTable weights) {
        this.weights = weights;
        this.wordCount = wordCount;
        setup(sc);
    }

    public void setup(SparkConf conf) {
        useAdaGrad = conf.getBoolean(ADAGRAD, false);
        negative = conf.getDouble(NEGATIVE, 5);
        numWords = conf.getInt(NUM_WORDS, 1);
        window = conf.getInt(WINDOW, 5);
        alpha = conf.getDouble(ALPHA, 0.025f);
        minAlpha = conf.getDouble(MIN_ALPHA, 1e-2f);
        totalWords = conf.getInt(NUM_WORDS, 1);
        iterations = conf.getInt(ITERATIONS, 5);
        vectorLength = conf.getInt(VECTOR_LENGTH, 100);

        initExpTable();

        if (negative > 0 && conf.contains(TABLE)) {
            try {
                ByteArrayInputStream bis = new ByteArrayInputStream(conf.get(TABLE).getBytes());
                DataInputStream dis = new DataInputStream(bis);
                table = Nd4j.read(dis);
            } catch (IOException e) {
                e.printStackTrace();
            }

        }

    }


    public int getVectorLength() {
        return vectorLength;
    }

    public void setVectorLength(int vectorLength) {
        this.vectorLength = vectorLength;
    }

    public double[] getExpTable() {
        return expTable;
    }

    public void setExpTable(double[] expTable) {
        this.expTable = expTable;
    }

    public InMemoryLookupTable getWeights() {
        return weights;
    }

    public void setWeights(InMemoryLookupTable weights) {
        this.weights = weights;
    }

    public Broadcast<AtomicLong> getWordCount() {
        return wordCount;
    }

    public void setWordCount(Broadcast<AtomicLong> wordCount) {
        this.wordCount = wordCount;
    }

    public int getLastChecked() {
        return lastChecked;
    }

    public void setLastChecked(int lastChecked) {
        this.lastChecked = lastChecked;
    }

    public static Logger getLog() {
        return log;
    }

    public int getIterations() {
        return iterations;
    }

    public void setIterations(int iterations) {
        this.iterations = iterations;
    }

    public int getTotalWords() {
        return totalWords;
    }

    public void setTotalWords(int totalWords) {
        this.totalWords = totalWords;
    }

    public double getMinAlpha() {
        return minAlpha;
    }

    public void setMinAlpha(double minAlpha) {
        this.minAlpha = minAlpha;
    }

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public AtomicLong getNextRandom() {
        return nextRandom;
    }

    public void setNextRandom(AtomicLong nextRandom) {
        this.nextRandom = nextRandom;
    }

    public int getWindow() {
        return window;
    }

    public void setWindow(int window) {
        this.window = window;
    }

    public INDArray getTable() {
        return table;
    }

    public void setTable(INDArray table) {
        this.table = table;
    }

    public int getNumWords() {
        return numWords;
    }

    public void setNumWords(int numWords) {
        this.numWords = numWords;
    }

    public double getNegative() {
        return negative;
    }

    public void setNegative(double negative) {
        this.negative = negative;
    }

    public boolean isUseAdaGrad() {
        return useAdaGrad;
    }

    public void setUseAdaGrad(boolean useAdaGrad) {
        this.useAdaGrad = useAdaGrad;
    }

    public static double getMAX_EXP() {
        return MAX_EXP;
    }

    public static void setMAX_EXP(double MAX_EXP) {
        Word2VecPerformerVoid.MAX_EXP = MAX_EXP;
    }

    /**
     * Train on a list of vocab words
     * @param sentence the list of vocab words to train on
     */
    public void trainSentence(final List<VocabWord> sentence, double alpha) {
        if (sentence != null && !sentence.isEmpty()) {
            for (int i = 0; i < sentence.size(); i++) {
                if (!sentence.get(i).getWord().endsWith("STOP")) {
                    nextRandom.set(nextRandom.get() * 25214903917L + 11);
                    skipGram(i, sentence, (int) nextRandom.get() % window, alpha);
                }
            }
        }

    }


    /**
     * Train via skip gram
     * @param i
     * @param sentence
     */
    public void skipGram(int i, List<VocabWord> sentence, int b, double alpha) {

        final VocabWord word = sentence.get(i);
        if (word != null && !sentence.isEmpty()) {
            int end = window * 2 + 1 - b;
            for (int a = b; a < end; a++) {
                if (a != window) {
                    int c = i - window + a;
                    if (c >= 0 && c < sentence.size()) {
                        VocabWord lastWord = sentence.get(c);
                        iterateSample(word, lastWord, alpha);
                    }
                }
            }
        }
    }



    /**
     * Iterate on the given 2 vocab words
     *
     * @param w1 the first word to iterate on
     * @param w2 the second word to iterate on
     */
    public void iterateSample(VocabWord w1, VocabWord w2, double alpha) {
        if (w2 == null || w2.getIndex() < 0)
            return;

        //current word vector
        INDArray l1 = weights.vector(w2.getWord());


        //error for current word and context
        INDArray neu1e = Nd4j.create(vectorLength);

        for (int i = 0; i < w1.getCodeLength(); i++) {
            int code = w1.getCodes().get(i);
            int point = w1.getPoints().get(i);

            INDArray syn1 = weights.getSyn1().slice(point);

            double dot = Nd4j.getBlasWrapper().dot(l1, syn1);

            if (dot >= -MAX_EXP && dot < MAX_EXP) {

                int idx = (int) ((dot + MAX_EXP) * ((double) expTable.length / MAX_EXP / 2.0));
                if (idx >= expTable.length)
                    continue;

                //score
                double f = expTable[idx];
                //gradient
                double g = (1 - code - f) * (useAdaGrad ? w1.getGradient(i, alpha, this.alpha) : alpha);


                if (neu1e.data().dataType() == DataBuffer.Type.DOUBLE) {
                    Nd4j.getBlasWrapper().axpy(g, syn1, neu1e);
                    Nd4j.getBlasWrapper().axpy(g, l1, syn1);
                } else {
                    Nd4j.getBlasWrapper().axpy((float) g, syn1, neu1e);
                    Nd4j.getBlasWrapper().axpy((float) g, l1, syn1);
                }
            }


        }


        //negative sampling
        if (negative > 0) {
            int target = w1.getIndex();
            int label;
            INDArray syn1Neg = weights.getSyn1Neg().slice(target);

            for (int d = 0; d < negative + 1; d++) {
                if (d == 0) {

                    label = 1;
                } else {
                    nextRandom.set(nextRandom.get() * 25214903917L + 11);
                    // FIXME: int cast
                    target = table.getInt((int) (nextRandom.get() >> 16) % (int) table.length());
                    if (target == 0)
                        target = (int) nextRandom.get() % (numWords - 1) + 1;
                    if (target == w1.getIndex())
                        continue;
                    label = 0;
                }

                double f = Nd4j.getBlasWrapper().dot(l1, syn1Neg);
                double g;
                if (f > MAX_EXP)
                    g = useAdaGrad ? w1.getGradient(target, (label - 1), this.alpha) : (label - 1) * alpha;
                else if (f < -MAX_EXP)
                    g = label * (useAdaGrad ? w1.getGradient(target, alpha, this.alpha) : alpha);
                else
                    g = useAdaGrad ? w1
                                    .getGradient(target,
                                                    label - expTable[(int) ((f + MAX_EXP)
                                                                    * (expTable.length / MAX_EXP / 2))],
                                                    this.alpha)
                                    : (label - expTable[(int) ((f + MAX_EXP) * (expTable.length / MAX_EXP / 2))])
                                                    * alpha;
                if (syn1Neg.data().dataType() == DataBuffer.Type.DOUBLE)
                    Nd4j.getBlasWrapper().axpy(g, neu1e, l1);
                else
                    Nd4j.getBlasWrapper().axpy((float) g, neu1e, l1);

                if (syn1Neg.data().dataType() == DataBuffer.Type.DOUBLE)
                    Nd4j.getBlasWrapper().axpy(g, syn1Neg, l1);
                else
                    Nd4j.getBlasWrapper().axpy((float) g, syn1Neg, l1);
            }
        }

        if (neu1e.data().dataType() == DataBuffer.Type.DOUBLE)
            Nd4j.getBlasWrapper().axpy(1.0, neu1e, l1);

        else
            Nd4j.getBlasWrapper().axpy(1.0f, neu1e, l1);

    }

    private void initExpTable() {
        for (int i = 0; i < expTable.length; i++) {
            double tmp = FastMath.exp((i / (double) expTable.length * 2 - 1) * MAX_EXP);
            expTable[i] = tmp / (tmp + 1.0);
        }
    }


    @Override
    public void call(Pair<List<VocabWord>, AtomicLong> pair) throws Exception {
        double numWordsSoFar = wordCount.getValue().doubleValue();

        List<VocabWord> sentence = pair.getFirst();
        double alpha2 = Math.max(minAlpha, alpha * (1 - (1.0 * numWordsSoFar / (double) totalWords)));
        int totalNewWords = 0;
        trainSentence(sentence, alpha2);
        totalNewWords += sentence.size();



        double newWords = totalNewWords + numWordsSoFar;
        double diff = Math.abs(newWords - lastChecked);
        if (diff >= 10000) {
            lastChecked = (int) newWords;
            log.info("Words so far " + newWords + " out of " + totalWords);
        }

        pair.getSecond().getAndAdd((long) totalNewWords);
    }


}
