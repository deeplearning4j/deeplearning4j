package org.deeplearning4j.datasets.iterator;

import lombok.NonNull;
import org.nd4j.linalg.primitives.Pair;

/**
 * @author raver119@gmail.com
 */
public class DoublesDataSetIterator extends AbstractDataSetIterator<double[]> {
    public DoublesDataSetIterator(@NonNull Iterable<Pair<double[], double[]>> iterable, int batchSize) {
        super(iterable, batchSize);
    }
}
