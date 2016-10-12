package org.nd4j.linalg.api.ops.aggregates;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * Aggregates are ops that work with custom operands, that are not limited to traditional X, Y and Z constraints.
 *
 * @author raver119@gmail.com
 */
public interface Aggregate {

    String name();

    int opNum();

    List<INDArray> getArguments();
}
