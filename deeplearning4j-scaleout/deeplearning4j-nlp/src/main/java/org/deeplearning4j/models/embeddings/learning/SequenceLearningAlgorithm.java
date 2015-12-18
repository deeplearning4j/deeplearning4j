package org.deeplearning4j.models.embeddings.learning;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Implementations of this interface should contain sequence-related learning algorithms. Like dbow or dm.
 *
 * @author raver119@gmail.com
 */
public interface SequenceLearningAlgorithm<T extends SequenceElement> {

    String getCodeName();

    void configure(VocabCache<T> vocabCache, WeightLookupTable<T> lookupTable, VectorsConfiguration configuration);

    void pretrain(SequenceIterator<T> iterator);

    void learnSequence(Sequence<T> sequence, AtomicLong nextRandom, double learningRate);

    boolean isEarlyTerminationHit();
}
