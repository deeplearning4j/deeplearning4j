package org.deeplearning4j.models.abstractvectors.transformers;

import org.deeplearning4j.models.abstractvectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.abstractvectors.sequence.Sequence;
import org.deeplearning4j.models.abstractvectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

/**
 *
 * @author raver119@gmail.com
 */
public interface SequenceTransformer<T extends SequenceElement, V extends Object> {

    VocabCache<T> derivedVocabulary();

    Sequence<T> transformToSequence(V object);
}
