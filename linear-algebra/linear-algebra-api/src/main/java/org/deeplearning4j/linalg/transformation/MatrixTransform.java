package org.deeplearning4j.linalg.transformation;

import com.google.common.base.Function;
import org.deeplearning4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

public interface MatrixTransform extends Function<INDArray,INDArray>,Serializable {

}
