package org.deeplearning4j.models.embeddings.training;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

/**
 * Implementations of this interface should contain element-related learning algorithms. Like skip-gram, cbow or glove
 *
 * @author raver119@gmail.com
 */
public interface ElementsLearningAlgorithm<T extends SequenceElement> {

    String getCodeName();

    void configure(VocabCache<T> vocabCache, WeightLookupTable<T> lookupTable);

    void learnSequence(Sequence<T> sequence);
}
