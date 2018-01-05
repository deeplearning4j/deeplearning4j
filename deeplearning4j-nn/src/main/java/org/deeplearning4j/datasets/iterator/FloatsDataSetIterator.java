package org.deeplearning4j.datasets.iterator;

import lombok.NonNull;
import org.nd4j.linalg.primitives.Pair;

/**
 * float[] wrapper for DataSetIterator impementation.
 *
 * This iterator creates DataSets out of externally-originated pairs of floats.
 *
 * @author raver119@gmail.com
 */
public class FloatsDataSetIterator extends AbstractDataSetIterator<float[]> {

    public FloatsDataSetIterator(@NonNull Iterable<Pair<float[], float[]>> iterable, int batchSize) {
        super(iterable, batchSize);
    }
}
