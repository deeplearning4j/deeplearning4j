package org.deeplearning4j.models.abstractvectors.iterators;

import lombok.NonNull;
import org.deeplearning4j.models.abstractvectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.abstractvectors.sequence.Sequence;
import org.deeplearning4j.models.abstractvectors.sequence.SequenceElement;



import java.util.Iterator;

/**
 * This is basic generic SequenceIterator implementation
 *
 * @author raver119@gmail.com
 */
public class AbstractSequenceIterator<T extends SequenceElement> implements SequenceIterator<T> {

    private Iterable<Sequence<T>> underlyingIterable;
    private Iterator<Sequence<T>> currentIterator;

    protected AbstractSequenceIterator(@NonNull Iterable<Sequence<T>> iterable) {
        this.underlyingIterable = iterable;
        this.currentIterator = iterable.iterator();
    }

    /**
     * Checks, if there's more sequences available
     * @return
     */
    @Override
    public boolean hasMoreSequences() {
        return currentIterator.hasNext();
    }

    /**
     * Returns next sequence out of iterator
     * @return
     */
    @Override
    public Sequence<T> nextSequence() {
        return currentIterator.next();
    }

    /**
     * Resets iterator to first position
     */
    @Override
    public void reset() {
        this.currentIterator = underlyingIterable.iterator();
    }

    public static class Builder<T extends SequenceElement> {
        private Iterable<Sequence<T>> underlyingIterable;

        /**
         * Builds AbstractSequenceIterator on top of Iterable object
         * @param iterable
         */
        public Builder(@NonNull Iterable<Sequence<T>> iterable) {
            this.underlyingIterable = iterable;
        }

        /**
         * Builds SequenceIterator
         * @return
         */
        public AbstractSequenceIterator<T> build() {
            AbstractSequenceIterator<T> iterator = new AbstractSequenceIterator<T>(underlyingIterable);

            return iterator;
        }
    }
}
