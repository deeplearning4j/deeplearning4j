package org.deeplearning4j.models.sequencevectors.interfaces;

import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 *
 * @author raver119@gmail.com
 */
public interface SequenceIterator<T extends SequenceElement> {

    boolean hasMoreSequences();

    Sequence<T> nextSequence();

    void reset();
}
