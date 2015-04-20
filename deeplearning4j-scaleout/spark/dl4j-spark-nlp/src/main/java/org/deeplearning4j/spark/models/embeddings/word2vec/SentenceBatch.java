package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author Adam Gibson
 */
public class SentenceBatch implements Function<Pair<List<VocabWord>,AtomicLong>,Word2VecChange> {

    private AtomicLong nextRandom = new AtomicLong(5);
    private Word2VecParam param;

    public SentenceBatch(Word2VecParam param) {
        this.param = param;
    }

    @Override
    public Word2VecChange call(Pair<List<VocabWord>, AtomicLong> tripleIterator) throws Exception {

        List<Triple<Integer,Integer,Integer>> changed = new ArrayList<>();
        Pair<List<VocabWord>, AtomicLong> next = tripleIterator;
        param.getWordsSeen().addAndGet(next.getSecond().get());
        double alpha = Math.max(param.getMinAlpha(), param.getAlpha() *
                (1 - (1.0 * param.getWordsSeen().get() / (double) param.getTotalWords())));
        trainSentence(next.getFirst(),alpha,changed);
        return new Word2VecChange(changed,param);
    }




    /**
     * Train on a list of vocab words
     * @param sentence the list of vocab words to train on
     */
    public void trainSentence(final List<VocabWord> sentence,double alpha,List<Triple<Integer,Integer,Integer>> changed) {
        if (sentence != null && !sentence.isEmpty()) {
            for (int i = 0; i < sentence.size(); i++) {
                if (!sentence.get(i).getWord().endsWith("STOP")) {
                    nextRandom.set(nextRandom.get() * 25214903917L + 11);
                    skipGram(i, sentence, (int) nextRandom.get() % param.getWindow(), alpha,changed);
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
    public void skipGram(int i,List<VocabWord> sentence, int b,double alpha,List<Triple<Integer,Integer,Integer>> changed) {

        final VocabWord word = sentence.get(i);
        int window = param.getWindow();
        if (word != null && !sentence.isEmpty()) {
            int end = window * 2 + 1 - b;
            for (int a = b; a < end; a++) {
                if (a != window) {
                    int c = i - window + a;
                    if (c >= 0 && c < sentence.size()) {
                        VocabWord lastWord = sentence.get(c);
                        iterateSample(word, lastWord, alpha,changed);
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
    public  void iterateSample(VocabWord w1, VocabWord w2,double alpha,List<Triple<Integer,Integer,Integer>> changed) {
        if(w2 == null || w2.getIndex() < 0 || w1.getIndex() == w2.getIndex())
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

        for(int i = 0; i < w1.getCodeLength(); i++) {
            int code = w1.getCodes().get(i);
            int point = w1.getPoints().get(i);

            INDArray syn1 = weights.getSyn1().slice(point);

            double dot = Nd4j.getBlasWrapper().dot(l1,syn1);

            if (dot >= -MAX_EXP && dot < MAX_EXP) {

                int idx = (int) ((dot + MAX_EXP) * ((double) expTable.length / MAX_EXP / 2.0));
                if (idx >= expTable.length)
                    continue;

                //score
                double f = expTable[idx];
                //gradient
                double g = (1 - code - f) * (useAdaGrad ? w1.getGradient(i, alpha) : alpha);


                if (neu1e.data().dataType() == DataBuffer.DOUBLE) {
                    Nd4j.getBlasWrapper().axpy(g, syn1, neu1e);
                    Nd4j.getBlasWrapper().axpy(g, l1, syn1);
                } else {
                    Nd4j.getBlasWrapper().axpy((float) g, syn1, neu1e);
                    Nd4j.getBlasWrapper().axpy((float) g, l1, syn1);
                }
            }

            changed.add(new Triple<>(w1.getIndex(), point, -1));

        }


        changed.add(new Triple<>(w1.getIndex(),w2.getIndex(),-1));
        //negative sampling
        if(negative > 0) {
            int target = w1.getIndex();
            int label;
            INDArray syn1Neg = weights.getSyn1Neg().slice(target);

            for (int d = 0; d < negative + 1; d++) {
                if (d == 0) {

                    label = 1;
                } else {
                    nextRandom.set(nextRandom.get() * 25214903917L + 11);
                    target = table.getInt((int) (nextRandom.get() >> 16) % table.length());
                    if (target == 0)
                        target = (int) nextRandom.get() % (numWords - 1) + 1;
                    if (target == w1.getIndex())
                        continue;
                    label = 0;
                }

                double f = Nd4j.getBlasWrapper().dot(l1, syn1Neg);
                double g;
                if (f > MAX_EXP)
                    g = useAdaGrad ? w1.getGradient(target, (label - 1)) : (label - 1) *  alpha;
                else if (f < -MAX_EXP)
                    g = label * (useAdaGrad ?  w1.getGradient(target, alpha) : alpha);
                else
                    g = useAdaGrad ? w1.getGradient(target, label - expTable[(int)((f + MAX_EXP) * (expTable.length / MAX_EXP / 2))]) : (label - expTable[(int)((f + MAX_EXP) * (expTable.length / MAX_EXP / 2))]) *   alpha;
                if(syn1Neg.data().dataType() == DataBuffer.DOUBLE)
                    Nd4j.getBlasWrapper().axpy(g,neu1e,l1);
                else
                    Nd4j.getBlasWrapper().axpy((float) g,neu1e,l1);

                if(syn1Neg.data().dataType() == DataBuffer.DOUBLE)
                    Nd4j.getBlasWrapper().axpy(g,syn1Neg,l1);
                else
                    Nd4j.getBlasWrapper().axpy((float) g,syn1Neg,l1);
                changed.add(new Triple<>(-1,-1,label));

            }
        }

        if(neu1e.data().dataType() == DataBuffer.DOUBLE)
            Nd4j.getBlasWrapper().axpy(1.0,neu1e,l1);

        else
            Nd4j.getBlasWrapper().axpy(1.0f,neu1e,l1);


    }

}
