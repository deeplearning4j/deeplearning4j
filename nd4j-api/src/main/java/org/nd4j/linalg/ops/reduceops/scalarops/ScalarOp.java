package org.nd4j.linalg.ops.reduceops.scalarops;

import com.google.common.base.Function;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A scalar op over an ndarray (iterates through the whole ndarray for an aggregate result)
 * @author Adam Gibson
 */
public interface ScalarOp extends Function<INDArray,Float> {
}
