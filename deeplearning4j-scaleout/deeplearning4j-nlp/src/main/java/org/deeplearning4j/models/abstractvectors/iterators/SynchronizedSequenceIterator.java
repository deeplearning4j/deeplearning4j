package org.deeplearning4j.models.abstractvectors.iterators;

import lombok.NonNull;
import org.deeplearning4j.models.abstractvectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.abstractvectors.sequence.Sequence;
import org.deeplearning4j.models.abstractvectors.sequence.SequenceElement;

/**
 * Synchronized version of AbstractSeuqenceIterator, implemented on top of it.
 * Suitable for cases with non-strict multithreading environment
 *
 * @author raver119@gmail.com
 */
public class SynchronizedSequenceIterator<T extends SequenceElement> implements SequenceIterator<T> {
    protected SequenceIterator<T> underlyingIterator;

    public SynchronizedSequenceIterator(@NonNull SequenceIterator<T> iterator) {
        this.underlyingIterator = iterator;
    }

    @Override
    public synchronized boolean hasMoreSequences() {
        return underlyingIterator.hasMoreSequences();
    }

    @Override
    public synchronized Sequence<T> nextSequence() {
        return underlyingIterator.nextSequence();
    }

    @Override
    public synchronized void reset() {
        underlyingIterator.reset();
    }
}
