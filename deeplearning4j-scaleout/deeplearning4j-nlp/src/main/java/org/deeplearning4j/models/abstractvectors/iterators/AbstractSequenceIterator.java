package org.deeplearning4j.models.abstractvectors.iterators;

import lombok.NonNull;
import org.deeplearning4j.models.abstractvectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.abstractvectors.sequence.Sequence;
import org.deeplearning4j.models.abstractvectors.sequence.SequenceElement;
import org.deeplearning4j.models.abstractvectors.transformers.SequenceTransformer;

/**
 * This is basic generic SequenceIterator implementation
 *
 * @author raver119@gmail.com
 */
public class AbstractSequenceIterator<T extends SequenceElement> implements SequenceIterator<T> {

    protected AbstractSequenceIterator() {

    }

    @Override
    public boolean hasMoreSequences() {
        return false;
    }

    @Override
    public Sequence<T> nextSequence() {
        return null;
    }

    @Override
    public void reset() {

    }

    public static class Builder<T extends SequenceElement> {


        /**
         * Builds AbstractSequenceIterator on top of Iterable object
         * @param iterable
         */
        public Builder(@NonNull Iterable<Sequence<T>> iterable) {

        }

        public Builder(@NonNull SequenceTransformer<T, ?> transformer) {

        }


        public AbstractSequenceIterator<T> build() {
            AbstractSequenceIterator<T> iterator = new AbstractSequenceIterator<T>();

            return iterator;
        }
    }
}
