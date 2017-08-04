package org.deeplearning4j.models.sequencevectors.transformers;

import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

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


    void reset();
}
