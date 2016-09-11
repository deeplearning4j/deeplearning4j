package org.deeplearning4j.models.embeddings.learning.impl.sequence;

import lombok.NonNull;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.learning.SequenceLearningAlgorithm;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class DBOW<T extends SequenceElement> implements SequenceLearningAlgorithm<T>{
    protected VocabCache<T> vocabCache;
    protected WeightLookupTable<T> lookupTable;
    protected VectorsConfiguration configuration;


    protected int window;
    protected boolean useAdaGrad;
    protected double negative;

    protected SkipGram<T> skipGram = new SkipGram<>();


    private static final Logger log = LoggerFactory.getLogger(DBOW.class);

    public DBOW() {

    }

    @Override
    public String getCodeName() {
        return "DBOW";
    }

    @Override
    public void configure(@NonNull VocabCache<T> vocabCache, @NonNull WeightLookupTable<T> lookupTable, @NonNull VectorsConfiguration configuration) {
        this.vocabCache = vocabCache;
        this.lookupTable = lookupTable;

        this.window = configuration.getWindow();
        this.useAdaGrad = configuration.isUseAdaGrad();
        this.negative = configuration.getNegative();

        skipGram.configure(vocabCache, lookupTable, configuration);
    }

    /**
     * DBOW doesn't involves any pretraining
     *
     * @param iterator
     */
    @Override
    public void pretrain(SequenceIterator<T> iterator) {

    }

    @Override
    public double learnSequence(@NonNull Sequence<T> sequence, @NonNull AtomicLong nextRandom, double learningRate) {
//        for(int i = 0; i < sequence.getElements().size(); i++) {
            dbow(0, sequence,  (int) nextRandom.get() % window, nextRandom, learningRate);
//        }

        return 0;
    }

    /**
     * DBOW has no reasons for early termination
     * @return
     */
    @Override
    public boolean isEarlyTerminationHit() {
        return false;
    }

    protected void dbow(int i, Sequence<T> sequence, int b, AtomicLong nextRandom, double alpha) {

        //final T word = sequence.getElements().get(i);
        List<T> sentence = skipGram.applySubsampling(sequence,nextRandom).getElements();

        List<T> labels = new ArrayList<>();
        labels.addAll(sequence.getSequenceLabels());

        if (sequence.getSequenceLabel() == null) throw new IllegalStateException("Label is NULL");

        if(sentence.isEmpty() || labels.isEmpty())
            return;

        for (T lastWord: labels) {
            for (T word:  sentence) {
                if (word == null) continue;

                skipGram.iterateSample(word, lastWord,nextRandom,alpha);
            }
        }
    }

    /**
     * This method does training on previously unseen paragraph, and returns inferred vector
     *
     * @param sequence
     * @param nextRandom
     * @param learningRate
     * @return
     */
    @Override
    public INDArray inferSequence(Sequence<T> sequence, long nextRandom, double learningRate) {
        throw new UnsupportedOperationException("not implemented for DBOW, please use DM instead");
    }
}
