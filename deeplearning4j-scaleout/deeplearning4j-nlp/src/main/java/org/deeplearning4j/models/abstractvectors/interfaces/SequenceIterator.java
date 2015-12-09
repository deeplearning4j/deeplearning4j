package org.deeplearning4j.models.abstractvectors.interfaces;

import org.deeplearning4j.models.abstractvectors.sequence.Sequence;
import org.deeplearning4j.models.abstractvectors.sequence.SequenceElement;

/**
 *
 * @author raver119@gmail.com
 */
public interface SequenceIterator<T extends SequenceElement> {

    boolean hasMoreSequences();

    Sequence<T> nextSequence();

    void reset();
}
