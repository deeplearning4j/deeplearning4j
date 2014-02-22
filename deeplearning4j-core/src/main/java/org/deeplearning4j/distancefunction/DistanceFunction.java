package org.deeplearning4j.distancefunction;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

import com.google.common.base.Function;

public interface DistanceFunction extends Function<DoubleMatrix,Double>,Serializable {

	

}
