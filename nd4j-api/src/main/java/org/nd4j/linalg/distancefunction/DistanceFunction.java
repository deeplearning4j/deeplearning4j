package org.nd4j.linalg.distancefunction;

import com.google.common.base.Function;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

public interface DistanceFunction extends Function<INDArray,Float>,Serializable {

	

}
