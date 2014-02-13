package com.ccc.deeplearning.transformation;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

import com.google.common.base.Function;

public interface MatrixTransform extends Function<DoubleMatrix,DoubleMatrix>,Serializable {

}
