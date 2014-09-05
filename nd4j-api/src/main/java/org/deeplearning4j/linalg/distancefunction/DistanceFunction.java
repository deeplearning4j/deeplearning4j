package org.deeplearning4j.linalg.distancefunction;

import com.google.common.base.Function;
import org.deeplearning4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

public interface DistanceFunction extends Function<INDArray,Double>,Serializable {

	

}
