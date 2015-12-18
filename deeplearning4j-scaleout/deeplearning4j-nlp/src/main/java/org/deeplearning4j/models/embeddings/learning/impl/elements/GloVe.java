package org.deeplearning4j.models.embeddings.learning.impl.elements;

import lombok.NonNull;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.glove.AbstractCoOccurrences;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

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
     * Learns sequence using SkipGram algorithm
     *
     * @param sequence
     * @param nextRandom
     * @param learningRate
     */
    @Override
    public void learnSequence(@NonNull Sequence<T> sequence, @NonNull AtomicLong nextRandom, double learningRate) {

    }
}
