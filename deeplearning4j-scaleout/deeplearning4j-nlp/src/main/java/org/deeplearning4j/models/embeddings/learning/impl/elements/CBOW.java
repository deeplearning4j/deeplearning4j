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
    public void learnSequence(Sequence<T> sequence, AtomicLong nextRandom, double learningRate) {
        Sequence<T> tempSequence = sequence;
        if (sampling > 0) tempSequence = applySubsampling(sequence, nextRandom);

        for (int i = 0; i < tempSequence.getElements().size(); i++) {
            nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
            cbow(i, tempSequence.getElements(),  (int) nextRandom.get() % window ,nextRandom, learningRate);
        }
    }

    @Override
    public boolean isEarlyTerminationHit() {
        return false;
    }

    public void iterateSample(int c, T word, AtomicLong nextRandom, double alpha ) {
        INDArray neu1e = Nd4j.zeros(lookupTable.layerSize());
    }

    public void cbow(int i, List<T> sentence, int b, AtomicLong nextRandom, double alpha) {
        int end =  window * 2 + 1 - b;
        int cw = 0;
        INDArray neu1 = Nd4j.zeros(lookupTable.layerSize());
        INDArray neu1e = Nd4j.zeros(lookupTable.layerSize());

        T currentWord = sentence.get(i);

        for(int a = b; a < end; a++) {
            if(a != window) {
                int c = i - window + a;
                if(c >= 0 && c < sentence.size()) {
                    T lastWord = sentence.get(c);

                    neu1.addiRowVector(syn0.getRow(lastWord.getIndex()));
                    cw++;
                }
            }

        }

        neu1.divi(cw);

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

            double g = (1 - code - f) * alpha;

            Nd4j.getBlasWrapper().level1().axpy(syn1row.length(),g, syn1row, neu1e);
            Nd4j.getBlasWrapper().level1().axpy(syn1row.length(),g, neu1, syn1row);
        }

        for(int a = b; a < end; a++) {
            if(a != window) {
                int c = i - window + a;
                if(c >= 0 && c < sentence.size()) {
                    T lastWord = sentence.get(c);

                    syn0.getRow(lastWord.getIndex()).addiRowVector(neu1e);
                    cw++;
                }
            }
        }

      //  logger.info("neu1: " + neu1);
      //  throw new RuntimeException();
    }

    protected Sequence<T> applySubsampling(@NonNull Sequence<T> sequence, @NonNull AtomicLong nextRandom) {
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
