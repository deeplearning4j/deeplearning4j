package com.ccc.deeplearning.nn.activation;

import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.util.MatrixUtil;

public class Sigmoid implements ActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6280602270833101092L;

	@Override
	public DoubleMatrix apply(DoubleMatrix arg0) {
		return MatrixUtil.sigmoid(arg0);
	}

	

}
