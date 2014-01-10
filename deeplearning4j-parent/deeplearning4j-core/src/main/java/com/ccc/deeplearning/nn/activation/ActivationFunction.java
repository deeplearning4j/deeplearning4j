package com.ccc.deeplearning.nn.activation;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

import com.google.common.base.Function;

public interface ActivationFunction extends Function<DoubleMatrix,DoubleMatrix>,Serializable {


}
