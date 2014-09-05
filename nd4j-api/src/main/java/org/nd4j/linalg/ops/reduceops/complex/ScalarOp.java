package org.nd4j.linalg.ops.reduceops.complex;

import com.google.common.base.Function;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * A scalar op over an ndarray (iterates through the whole ndarray for an aggregate result)
 * @author Adam Gibson
 */
public interface ScalarOp extends Function<IComplexNDArray,IComplexNumber> {
}
