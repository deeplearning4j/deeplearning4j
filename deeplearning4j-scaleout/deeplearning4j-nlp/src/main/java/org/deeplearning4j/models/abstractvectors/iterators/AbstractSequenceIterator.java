package org.deeplearning4j.models.abstractvectors.iterators;

import lombok.NonNull;
import org.deeplearning4j.models.abstractvectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.abstractvectors.sequence.Sequence;
import org.deeplearning4j.models.abstractvectors.sequence.SequenceElement;
import org.deeplearning4j.models.abstractvectors.transformers.SequenceTransformer;
import org.deeplearning4j.models.abstractvectors.transformers.TransformerFactory;

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

    @Override
    public boolean hasMoreSequences() {
        return currentIterator.hasNext();
    }

    @Override
    public Sequence<T> nextSequence() {
        return currentIterator.next();
    }

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

        /*
        public Builder(@NonNull SequenceTransformer<T, ?> transformer) {

        }
*/

        public AbstractSequenceIterator<T> build() {
            AbstractSequenceIterator<T> iterator = new AbstractSequenceIterator<T>(underlyingIterable);

            return iterator;
        }
    }
}
