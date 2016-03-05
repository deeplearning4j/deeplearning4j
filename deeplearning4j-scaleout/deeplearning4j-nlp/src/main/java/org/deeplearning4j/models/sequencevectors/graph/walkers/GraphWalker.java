package org.deeplearning4j.models.sequencevectors.graph.walkers;

import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * @author raver119@gmail.com
 */
public interface GraphWalker<T extends SequenceElement> {

    boolean hasNext();

    Sequence<T> next();

    void reset();
}
