package org.deeplearning4j.models.embeddings.learning.impl.elements;

import lombok.NonNull;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.glove.AbstractCoOccurrences;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * GloVe implementation for SequenceVectors
 *
 * @author raver119@gmail.com
 */
public  class GloVe<T extends SequenceElement> implements ElementsLearningAlgorithm<T> {

    private VocabCache<T> vocabCache;
    private AbstractCoOccurrences<T> coOccurrences;
    private WeightLookupTable<T> lookupTable;
    private VectorsConfiguration configuration;

    @Override
    public String getCodeName() {
        return "GloVe";
    }

    @Override
    public void configure(@NonNull VocabCache<T> vocabCache, @NonNull WeightLookupTable<T> lookupTable, @NonNull VectorsConfiguration configuration) {
        this.vocabCache = vocabCache;
        this.lookupTable = lookupTable;
        this.configuration = configuration;
    }

    /**
     * pretrain is used to build CoOccurrence matrix for GloVe algorithm
     * @param iterator
     */
    @Override
    public void pretrain(@NonNull SequenceIterator<T> iterator) {
        // CoOccurence table should be built here
        coOccurrences = new AbstractCoOccurrences.Builder<T>()
                // TODO: symmetric should be handled via VectorsConfiguration
                .symmetric(false)
                .windowSize(configuration.getWindow())
                .iterate(iterator)
                .workers(Runtime.getRuntime().availableProcessors())
                .vocabCache(vocabCache)
                .build();

        coOccurrences.fit();
    }

    /**
     * Learns sequence using GloVe algorithm
     *
     * @param sequence
     * @param nextRandom
     * @param learningRate
     */
    @Override
    public void learnSequence(@NonNull Sequence<T> sequence, @NonNull AtomicLong nextRandom, double learningRate) {
        /*
                GloVe learning algorithm is implemented like a hack, over existing code base. It's called in SequenceVectors context, but actually only for the first call.
                All subsequent calls will met early termination condition, and will be successfully ignored. But since elements vectors will be updated within first call,
                this will allow compatibility with everything beyond this implementaton
         */
    }

    /**
     *  Since GloVe is learning representations using elements CoOccurences, all training is done in GloVe class internally, so only first thread will execute learning process,
     *  and the rest of parent threads will just exit learning process
     *
     * @return True, if training should stop, False otherwise.
     */
    @Override
    public boolean isEarlyTerminationHit() {
        return false;
    }

    private class GloveCalculationsThread extends Thread implements Runnable {
        private final int threadId;
        private final AbstractCoOccurrences<T> coOccurrences;

        public GloveCalculationsThread(int threadId, @NonNull AbstractCoOccurrences<T> coOccurrences) {
            this.threadId = threadId;
            this.coOccurrences = coOccurrences;

            this.setName("GloVe ElementsLearningAlgorithm thread " + this.threadId);
        }

        @Override
        public void run() {
            List<Pair<T, T>> coList = coOccurrences.coOccurrenceList();
            for (int x = 0; x < threadId; x++) {
                // no for each pair do appropriate training
                T element1 = coList.get(x).getFirst();
                T element2 = coList.get(x).getFirst();
                //double weight = coOccurrences.getCoOccurence
            }
        }
    }
}
