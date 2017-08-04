package org.deeplearning4j.models.sequencevectors.iterators;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.util.Iterator;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This is basic generic SequenceIterator implementation
 *
 * @author raver119@gmail.com
 */
public class AbstractSequenceIterator<T extends SequenceElement> implements SequenceIterator<T> {

    private Iterable<Sequence<T>> underlyingIterable;
    private Iterator<Sequence<T>> currentIterator;

    // used to tag each sequence with own Id
    protected AtomicInteger tagger = new AtomicInteger(0);

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
        Sequence<T> sequence = currentIterator.next();
        sequence.setSequenceId(tagger.getAndIncrement());
        return sequence;
    }

    /**
     * Resets iterator to first position
     */
    @Override
    public void reset() {
        tagger.set(0);
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
            AbstractSequenceIterator<T> iterator = new AbstractSequenceIterator<>(underlyingIterable);

            return iterator;
        }
    }
}
