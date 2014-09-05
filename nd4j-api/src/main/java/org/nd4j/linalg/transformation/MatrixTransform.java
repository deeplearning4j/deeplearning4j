package org.nd4j.linalg.transformation;

import com.google.common.base.Function;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

public interface MatrixTransform extends Function<INDArray,INDArray>,Serializable {

}
