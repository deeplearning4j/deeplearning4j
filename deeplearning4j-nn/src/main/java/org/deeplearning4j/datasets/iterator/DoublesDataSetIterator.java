package org.deeplearning4j.datasets.iterator;

import lombok.NonNull;
import org.deeplearning4j.berkeley.Pair;

/**
 * @author raver119@gmail.com
 */
public class DoublesDataSetIterator extends AbstractDataSetIterator<double[]> {
    public DoublesDataSetIterator(@NonNull Iterable<Pair<double[], double[]>> iterable, int batchSize) {
        super(iterable, batchSize);
    }
}
