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

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Triple;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author Adam Gibson
 */
@Deprecated
public class SentenceBatch implements Function<Word2VecFuncCall, Word2VecChange> {

    private AtomicLong nextRandom = new AtomicLong(5);
    //    private static Logger log = LoggerFactory.getLogger(SentenceBatch.class);


    @Override
    public Word2VecChange call(Word2VecFuncCall sentence) throws Exception {
        Word2VecParam param = sentence.getParam().getValue();
        List<Triple<Integer, Integer, Integer>> changed = new ArrayList<>();
        double alpha = Math.max(param.getMinAlpha(),
                        param.getAlpha() * (1 - (1.0 * sentence.getWordsSeen() / (double) param.getTotalWords())));

        trainSentence(param, sentence.getSentence(), alpha, changed);
        return new Word2VecChange(changed, param);
    }


    /**
     * Train on a list of vocab words
     * @param sentence the list of vocab words to train on
     */
    public void trainSentence(Word2VecParam param, final List<VocabWord> sentence, double alpha,
                    List<Triple<Integer, Integer, Integer>> changed) {
        if (sentence != null && !sentence.isEmpty()) {
            for (int i = 0; i < sentence.size(); i++) {
                VocabWord vocabWord = sentence.get(i);
                if (vocabWord != null && vocabWord.getWord().endsWith("STOP")) {
                    nextRandom.set(nextRandom.get() * 25214903917L + 11);
                    skipGram(param, i, sentence, (int) nextRandom.get() % param.getWindow(), alpha, changed);
                }
            }
        }
    }


    /**
     * Train via skip gram
     * @param i the current word
     * @param sentence the sentence to train on
     * @param b
     * @param alpha the learning rate
     */
    public void skipGram(Word2VecParam param, int i, List<VocabWord> sentence, int b, double alpha,
                    List<Triple<Integer, Integer, Integer>> changed) {

        final VocabWord word = sentence.get(i);
        int window = param.getWindow();
        if (word != null && !sentence.isEmpty()) {
            int end = window * 2 + 1 - b;
            for (int a = b; a < end; a++) {
                if (a != window) {
                    int c = i - window + a;
                    if (c >= 0 && c < sentence.size()) {
                        VocabWord lastWord = sentence.get(c);
                        iterateSample(param, word, lastWord, alpha, changed);
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
    public void iterateSample(Word2VecParam param, VocabWord w1, VocabWord w2, double alpha,
                    List<Triple<Integer, Integer, Integer>> changed) {
        if (w2 == null || w2.getIndex() < 0 || w1.getIndex() == w2.getIndex() || w1.getWord().equals("STOP")
                        || w2.getWord().equals("STOP") || w1.getWord().equals("UNK") || w2.getWord().equals("UNK"))
            return;
        int vectorLength = param.getVectorLength();
        InMemoryLookupTable weights = param.getWeights();
        boolean useAdaGrad = param.isUseAdaGrad();
        double negative = param.getNegative();
        INDArray table = param.getTable();
        double[] expTable = param.getExpTable().getValue();
        double MAX_EXP = 6;
        int numWords = param.getNumWords();
        //current word vector
        INDArray l1 = weights.vector(w2.getWord());


        //error for current word and context
        INDArray neu1e = Nd4j.create(vectorLength);

        for (int i = 0; i < w1.getCodeLength(); i++) {
            int code = w1.getCodes().get(i);
            int point = w1.getPoints().get(i);

            INDArray syn1 = weights.getSyn1().slice(point);

            double dot = Nd4j.getBlasWrapper().level1().dot(syn1.length(), 1.0, l1, syn1);

            if (dot < -MAX_EXP || dot >= MAX_EXP)
                continue;

            int idx = (int) ((dot + MAX_EXP) * ((double) expTable.length / MAX_EXP / 2.0));

            //score
            double f = expTable[idx];
            //gradient
            double g = (1 - code - f) * (useAdaGrad ? w1.getGradient(i, alpha, alpha) : alpha);


            Nd4j.getBlasWrapper().level1().axpy(syn1.length(), g, syn1, neu1e);
            Nd4j.getBlasWrapper().level1().axpy(syn1.length(), g, l1, syn1);


            changed.add(new Triple<>(point, w1.getIndex(), -1));

        }


        changed.add(new Triple<>(w1.getIndex(), w2.getIndex(), -1));
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
                    g = useAdaGrad ? w1.getGradient(target, (label - 1), alpha) : (label - 1) * alpha;
                else if (f < -MAX_EXP)
                    g = label * (useAdaGrad ? w1.getGradient(target, alpha, alpha) : alpha);
                else
                    g = useAdaGrad ? w1
                                    .getGradient(target,
                                                    label - expTable[(int) ((f + MAX_EXP)
                                                                    * (expTable.length / MAX_EXP / 2))],
                                                    alpha)
                                    : (label - expTable[(int) ((f + MAX_EXP) * (expTable.length / MAX_EXP / 2))])
                                                    * alpha;
                Nd4j.getBlasWrapper().level1().axpy(l1.length(), g, neu1e, l1);

                Nd4j.getBlasWrapper().level1().axpy(l1.length(), g, syn1Neg, l1);

                changed.add(new Triple<>(-1, -1, label));

            }
        }


        Nd4j.getBlasWrapper().level1().axpy(l1.length(), 1.0f, neu1e, l1);


    }

}
