package org.deeplearning4j.models.abstractvectors.transformers;

import org.deeplearning4j.models.abstractvectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.abstractvectors.sequence.Sequence;
import org.deeplearning4j.models.abstractvectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import java.util.Iterator;

/**
 *
 * @author raver119@gmail.com
 */
public interface SequenceTransformer<T extends SequenceElement, V extends Object> {

    /**
     * Returns Vocabulary derived from underlying data source.
     * In default implementations this method heavily relies on transformToSequence() method.
     *
     * @return
     */
    //VocabCache<T> derivedVocabulary();

    /**
     * This is generic method for transformation data from any format to Sequence of SequenceElement.
     * It will be used both in Vocab building, and in training process
     *
     * @param object - Object to be transformed into Sequence
     * @return
     */
    Sequence<T> transformToSequence(V object);
}
