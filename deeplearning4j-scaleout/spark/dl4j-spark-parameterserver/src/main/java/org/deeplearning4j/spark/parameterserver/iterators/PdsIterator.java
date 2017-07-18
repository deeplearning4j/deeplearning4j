package org.deeplearning4j.spark.parameterserver.iterators;

import lombok.NonNull;
import org.apache.spark.input.PortableDataStream;
import org.deeplearning4j.spark.parameterserver.callbacks.PortableDataStreamCallback;
import org.nd4j.linalg.dataset.DataSet;

import java.util.Iterator;
import java.util.function.Consumer;

/**
 * @author raver119@gmail.com
 */
public class PdsIterator implements Iterator<DataSet> {
    protected final Iterator<PortableDataStream> iterator;
    protected final PortableDataStreamCallback callback;

    public PdsIterator(@NonNull Iterator<PortableDataStream> pds, @NonNull PortableDataStreamCallback callback) {
        this.iterator = pds;
        this.callback = callback;
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public DataSet next() {
        return callback.compute(iterator.next());
    }

    @Override
    public void remove() {
        // no-op
    }

    @Override
    public void forEachRemaining(Consumer<? super DataSet> action) {
        throw new UnsupportedOperationException();
    }
}
