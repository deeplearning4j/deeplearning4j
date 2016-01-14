package org.deeplearning4j.models.embeddings.learning.impl.elements;

import lombok.NonNull;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Skip-Gram implementation for dl4j SequenceVectors
 *
 * @author raver119@gmail.com
 */
public class SkipGram<T extends SequenceElement> implements ElementsLearningAlgorithm<T> {
    protected VocabCache<T> vocabCache;
    protected WeightLookupTable<T> lookupTable;
    protected VectorsConfiguration configuration;

    protected static double MAX_EXP = 6;
    protected double[] expTable;

    protected int window;
    protected boolean useAdaGrad;
    protected double negative;
    protected INDArray syn0, syn1, syn1Neg, table;

    /**
     * Dummy construction is required for reflection
     */
    public SkipGram() {

    }

    /**
     * Returns implementation code name
     *
     * @return
     */
    @Override
    public String getCodeName() {
        return "SkipGram";
    }

    /**
     * SkipGram initialization over given vocabulary and WeightLookupTable
     *
     * @param vocabCache
     * @param lookupTable
     * @param configuration
     */
    @Override
    public void configure(@NonNull VocabCache<T> vocabCache, @NonNull WeightLookupTable<T> lookupTable, @NonNull VectorsConfiguration configuration) {
        this.vocabCache = vocabCache;
        this.lookupTable = lookupTable;
        this.configuration = configuration;

        this.expTable = ((InMemoryLookupTable<T>) lookupTable).getExpTable();
        this.syn0 = ((InMemoryLookupTable<T>) lookupTable).getSyn0();
        this.syn1 = ((InMemoryLookupTable<T>) lookupTable).getSyn1();
        this.syn1Neg = ((InMemoryLookupTable<T>) lookupTable).getSyn1Neg();
        this.table = ((InMemoryLookupTable<T>) lookupTable).getTable();

        this.window = configuration.getWindow();
        this.useAdaGrad = configuration.isUseAdaGrad();
        this.negative = configuration.getNegative();
    }

    /**
     * SkipGram doesn't involves any pretraining
     *
     * @param iterator
     */
    @Override
    public void pretrain(SequenceIterator<T> iterator) {

    }

    /**
     * Learns sequence using SkipGram algorithm
     *
     * @param sequence
     * @param nextRandom
     * @param learningRate
     */
    @Override
    public void learnSequence(@NonNull Sequence<T> sequence, @NonNull AtomicLong nextRandom, @NonNull double learningRate) {
        for(int i = 0; i < sequence.getElements().size(); i++) {
            nextRandom.set(nextRandom.get() * 25214903917L + 11);
            skipGram(i, sequence.getElements(), (int) nextRandom.get() % window ,nextRandom, learningRate);
        }
    }

    /**
     * SkipGram has no reasons for early termination ever.
     *
     * @return
     */
    @Override
    public boolean isEarlyTerminationHit() {
        return false;
    }

    private void skipGram(int i, List<T> sentence, int b, AtomicLong nextRandom, double alpha) {
        final T word = sentence.get(i);
        if(word == null || sentence.isEmpty())
            return;

        int end =  window * 2 + 1 - b;
        for(int a = b; a < end; a++) {
            if(a != window) {
                int c = i - window + a;
                if(c >= 0 && c < sentence.size()) {
                    T lastWord = sentence.get(c);
                    iterateSample(word,lastWord,nextRandom,alpha);
                }
            }
        }
    }

    public  void iterateSample(T w1, T w2,AtomicLong nextRandom,double alpha) {
        if(w2 == null || w2.getIndex() < 0 || w1.getIndex() == w2.getIndex() || w1.getLabel().equals("STOP") || w2.getLabel().equals("STOP") || w1.getLabel().equals("UNK") || w2.getLabel().equals("UNK"))
            return;
        //current word vector
        INDArray l1 = this.syn0.slice(w2.getIndex());


        //error for current word and context
        INDArray neu1e = Nd4j.create(configuration.getLayersSize());

      //  System.out.println("--------------------------");
        for(int i = 0; i < w1.getCodeLength(); i++) {
            int code = w1.getCodes().get(i);
            int point = w1.getPoints().get(i);
            if(point >= syn0.rows() || point < 0)
                throw new IllegalStateException("Illegal point " + point);
            //other word vector

            INDArray syn1 = this.syn1.slice(point);


            double dot = Nd4j.getBlasWrapper().dot(l1,syn1);

            if(dot < -MAX_EXP || dot >= MAX_EXP)
                continue;


            int idx = (int) ((dot + MAX_EXP) * ((double) expTable.length / MAX_EXP / 2.0));
            if(idx >= expTable.length)
                continue;

            //score
            double f =  expTable[idx];
            //gradient
            double g = useAdaGrad ?  w1.getGradient(i, (1 - code - f), alpha) : (1 - code - f) * alpha;

            if(neu1e.data().dataType() == DataBuffer.Type.FLOAT) {
                Nd4j.getBlasWrapper().level1().axpy(syn1.length(), g, syn1, neu1e);
                Nd4j.getBlasWrapper().level1().axpy(syn1.length(), g, l1, syn1);

            }

            else {
                Nd4j.getBlasWrapper().level1().axpy(syn1.length(),g, syn1, neu1e);
                Nd4j.getBlasWrapper().level1().axpy(syn1.length(),g, l1, syn1);

            }
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
                    int idx = Math.abs((int) (nextRandom.get() >> 16) % table.length());

                    target = table.getInt(idx);
                    if (target <= 0)
                        target = (int) nextRandom.get() % (vocabCache.numWords() - 1) + 1;

                    if (target == w1.getIndex())
                        continue;
                    label = 0;
                }


                if(target >= syn1Neg.rows() || target < 0)
                    continue;

                double f = Nd4j.getBlasWrapper().dot(l1,syn1Neg.slice(target));
                double g;
                if (f > MAX_EXP)
                    g = useAdaGrad ? lookupTable.getGradient(target, (label - 1)) : (label - 1) *  alpha;
                else if (f < -MAX_EXP)
                    g = label * (useAdaGrad ?  lookupTable.getGradient(target, alpha) : alpha);
                else {
                    int idx = (int) ((f + MAX_EXP) * (expTable.length / MAX_EXP / 2));
                    if (idx >= expTable.length)
                        continue;

                    g = useAdaGrad ? lookupTable.getGradient(target, label - expTable[idx]) : (label - expTable[idx]) * alpha;
                }
                if(syn0.data().dataType() == DataBuffer.Type.DOUBLE)
                    Nd4j.getBlasWrapper().axpy(g,syn1Neg.slice(target),neu1e);
                else
                    Nd4j.getBlasWrapper().axpy((float) g,syn1Neg.slice(target),neu1e);

                if(syn0.data().dataType() == DataBuffer.Type.DOUBLE)
                    Nd4j.getBlasWrapper().axpy(g,l1,syn1Neg.slice(target));
                else
                    Nd4j.getBlasWrapper().axpy((float) g,l1,syn1Neg.slice(target));
            }

        if(syn0.data().dataType() == DataBuffer.Type.DOUBLE)
            Nd4j.getBlasWrapper().axpy(1.0,neu1e,l1);

        else
            Nd4j.getBlasWrapper().axpy(1.0f,neu1e,l1);
    }
}
