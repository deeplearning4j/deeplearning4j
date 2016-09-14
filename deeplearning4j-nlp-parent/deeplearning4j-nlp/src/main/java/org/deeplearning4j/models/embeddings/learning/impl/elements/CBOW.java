package org.deeplearning4j.models.embeddings.learning.impl.elements;

import lombok.NonNull;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * CBOW implementation for DeepLearning4j
 *
 * @author raver119@gmail.com
 */
public class CBOW<T extends SequenceElement> implements ElementsLearningAlgorithm<T>{
    private VocabCache<T> vocabCache;
    private WeightLookupTable<T> lookupTable;
    private VectorsConfiguration configuration;

    private static final Logger logger = LoggerFactory.getLogger(CBOW.class);

    protected static double MAX_EXP = 6;

    protected int window;
    protected boolean useAdaGrad;
    protected double negative;
    protected double sampling;
    protected int[] variableWindows;

    protected double[] expTable;

    protected INDArray syn0, syn1, syn1Neg, table;

    @Override
    public String getCodeName() {
        return "CBOW";
    }

    @Override
    public void configure(@NonNull VocabCache<T> vocabCache, @NonNull WeightLookupTable<T> lookupTable, @NonNull VectorsConfiguration configuration) {
        this.vocabCache = vocabCache;
        this.lookupTable = lookupTable;
        this.configuration = configuration;

        this.window = configuration.getWindow();
        this.useAdaGrad = configuration.isUseAdaGrad();
        this.negative = configuration.getNegative();
        this.sampling = configuration.getSampling();

        this.syn0 = ((InMemoryLookupTable<T>) lookupTable).getSyn0();
        this.syn1 = ((InMemoryLookupTable<T>) lookupTable).getSyn1();
        this.syn1Neg = ((InMemoryLookupTable<T>) lookupTable).getSyn1Neg();
        this.expTable = ((InMemoryLookupTable<T>) lookupTable).getExpTable();
        this.table = ((InMemoryLookupTable<T>) lookupTable).getTable();
        this.variableWindows = configuration.getVariableWindows();
    }

    /**
     * CBOW doesn't involve any pretraining
     *
     * @param iterator
     */
    @Override
    public void pretrain(SequenceIterator<T> iterator) {
        // no-op
    }

    @Override
    public double learnSequence(Sequence<T> sequence, AtomicLong nextRandom, double learningRate) {
        Sequence<T> tempSequence = sequence;
        if (sampling > 0) tempSequence = applySubsampling(sequence, nextRandom);

        int currentWindow = window;

        if (variableWindows != null && variableWindows.length != 0) {
            currentWindow = variableWindows[RandomUtils.nextInt(variableWindows.length)];
        }

        for (int i = 0; i < tempSequence.getElements().size(); i++) {
            nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
            cbow(i, tempSequence.getElements(),  (int) nextRandom.get() % currentWindow ,nextRandom, learningRate, currentWindow);
        }

        return 0;
    }

    @Override
    public boolean isEarlyTerminationHit() {
        return false;
    }

    public INDArray iterateSample(T currentWord, INDArray neu1, AtomicLong nextRandom, double alpha, boolean isInference) {
        INDArray neu1e = Nd4j.zeros(lookupTable.layerSize());

        for (int p = 0; p < currentWord.getCodeLength(); p++) {
            double f = 0;
            int code = currentWord.getCodes().get(p);
            int point = currentWord.getPoints().get(p);

            INDArray syn1row = syn1.getRow(point);

            double dot = Nd4j.getBlasWrapper().dot(neu1, syn1.getRow(point));

            if(dot < -MAX_EXP || dot >= MAX_EXP)
                continue;

            int idx = (int) ((dot + MAX_EXP) * ((double) expTable.length / MAX_EXP / 2.0));
            if(idx >= expTable.length)
                continue;

            //score
            f =  expTable[idx];

            double g = useAdaGrad ?  currentWord.getGradient(p, (1 - code - f), alpha) : (1 - code - f) * alpha;

            Nd4j.getBlasWrapper().level1().axpy(syn1row.length(),g, syn1row, neu1e);
            if (!isInference)
                Nd4j.getBlasWrapper().level1().axpy(syn1row.length(),g, neu1, syn1row);
            else
                Nd4j.getBlasWrapper().level1().axpy(syn1row.length(),g, neu1, syn1row.dup());
        }

        if (negative > 0 && !isInference) {
            int target = currentWord.getIndex();
            int label;

            for (int d = 0; d < negative + 1; d++) {
                if (d == 0)
                    label = 1;
                else {
                    nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
                    int idx = Math.abs((int) (nextRandom.get() >> 16) % table.length());

                    target = table.getInt(idx);
                    if (target <= 0)
                        target = (int) nextRandom.get() % (vocabCache.numWords() - 1) + 1;

                    if (target == currentWord.getIndex())
                        continue;
                    label = 0;
                }


                if(target >= syn1Neg.rows() || target < 0)
                    continue;

                double f = Nd4j.getBlasWrapper().dot(neu1,syn1Neg.slice(target));
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

                Nd4j.getBlasWrapper().level1().axpy(lookupTable.layerSize(), g, syn1Neg.slice(target),neu1e);
                Nd4j.getBlasWrapper().level1().axpy(lookupTable.layerSize(), g, neu1,syn1Neg.slice(target));
            }
        }

     //   Nd4j.getBlasWrapper().level1().axpy(lookupTable.layerSize(), 1.0, neu1e, neu1);

        return neu1e;
    }

    public void cbow(int i, List<T> sentence, int b, AtomicLong nextRandom, double alpha, int currentWindow) {
        int end =  window * 2 + 1 - b;
        int cw = 0;
        INDArray neu1 = Nd4j.zeros(lookupTable.layerSize());

        T currentWord = sentence.get(i);

        for(int a = b; a < end; a++) {
            if(a != currentWindow) {
                int c = i - currentWindow + a;
                if(c >= 0 && c < sentence.size()) {
                    T lastWord = sentence.get(c);

                    neu1.addiRowVector(syn0.getRow(lastWord.getIndex()));
                    cw++;
                }
            }
        }

        if (cw == 0)
            return;

        neu1.divi(cw);

        INDArray neu1e = iterateSample(currentWord, neu1, nextRandom, alpha, false);

        for(int a = b; a < end; a++) {
            if(a != window) {
                int c = i - window + a;
                if(c >= 0 && c < sentence.size()) {
                    T lastWord = sentence.get(c);
                    INDArray syn0row = syn0.getRow(lastWord.getIndex());
                    Nd4j.getBlasWrapper().level1().axpy(lookupTable.layerSize(), 1.0, neu1e, syn0row);
                }
            }
        }
    }

    public Sequence<T> applySubsampling(@NonNull Sequence<T> sequence, @NonNull AtomicLong nextRandom) {
        Sequence<T> result = new Sequence<>();

        // subsampling implementation, if subsampling threshold met, just continue to next element
        if (sampling > 0) {
            result.setSequenceId(sequence.getSequenceId());
            if (sequence.getSequenceLabels() != null) result.setSequenceLabels(sequence.getSequenceLabels());
            if (sequence.getSequenceLabel() != null) result.setSequenceLabel(sequence.getSequenceLabel());

            for (T element : sequence.getElements()) {
                double numWords = vocabCache.totalWordOccurrences();
                double ran = (Math.sqrt(element.getElementFrequency() / (sampling * numWords)) + 1) * (sampling * numWords) / element.getElementFrequency();

                nextRandom.set(nextRandom.get() * 25214903917L + 11);

                if (ran < (nextRandom.get() & 0xFFFF) / (double) 65536) {
                    continue;
                }
                result.addElement(element);
            }
            return result;
        } else return sequence;
    }
}
