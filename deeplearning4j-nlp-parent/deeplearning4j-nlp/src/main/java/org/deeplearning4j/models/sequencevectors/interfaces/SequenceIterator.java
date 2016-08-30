package org.deeplearning4j.models.sequencevectors.interfaces;

import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * SequenceIterator is basic interface for learning abstract data that can be represented as sequence of some elements.
 *
 * @author raver119@gmail.com
 */
public interface SequenceIterator<T extends SequenceElement> {

    boolean hasMoreSequences();

    Sequence<T> nextSequence();

    void reset();
}
