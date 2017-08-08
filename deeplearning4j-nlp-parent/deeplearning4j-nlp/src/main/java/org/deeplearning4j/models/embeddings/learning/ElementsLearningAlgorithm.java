package org.deeplearning4j.models.embeddings.learning;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Implementations of this interface should contain element-related learning algorithms. Like skip-gram, cbow or glove
 *
 * @author raver119@gmail.com
 */
public interface ElementsLearningAlgorithm<T extends SequenceElement> {

    String getCodeName();

    void configure(VocabCache<T> vocabCache, WeightLookupTable<T> lookupTable, VectorsConfiguration configuration);

    void pretrain(SequenceIterator<T> iterator);

    /**
     * This method does training over the sequence of elements passed into it
     *
     * @param sequence
     * @param nextRandom
     * @param learningRate
     * @return average score for this sequence
     */
    double learnSequence(Sequence<T> sequence, AtomicLong nextRandom, double learningRate);

    boolean isEarlyTerminationHit();

    void finish();
}
