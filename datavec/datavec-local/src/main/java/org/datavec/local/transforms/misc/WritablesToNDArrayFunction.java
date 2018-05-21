package org.datavec.local.transforms.misc;

import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.List;

/**
 * Function for converting lists of Writables to a single
 * NDArray row vector. Necessary for creating and saving a
 * dense matrix representation of raw data.
 *
 * @author dave@skymind.io
 */
public class WritablesToNDArrayFunction implements Function<List<Writable>, INDArray> {

    @Override
    public INDArray apply(List<Writable> c) {
        int length = 0;
        for (Writable w : c) {
            if (w instanceof NDArrayWritable) {
                INDArray a = ((NDArrayWritable) w).get();
                if (a.isRowVector()) {
                    length += a.columns();
                } else {
                    throw new UnsupportedOperationException("NDArrayWritable is not a row vector."
                                    + " Can only concat row vectors with other writables. Shape: "
                                    + Arrays.toString(a.shape()));
                }
            } else {
                length++;
            }
        }

        INDArray arr = Nd4j.zeros(length);
        int idx = 0;
        for (Writable w : c) {
            if (w instanceof NDArrayWritable) {
                INDArray subArr = ((NDArrayWritable) w).get();
                int subLength = subArr.columns();
                arr.get(NDArrayIndex.interval(idx, idx + subLength)).assign(subArr);
                idx += subLength;
            } else {
                arr.putScalar(idx++, w.toDouble());
            }
        }

        return arr;
    }
}
