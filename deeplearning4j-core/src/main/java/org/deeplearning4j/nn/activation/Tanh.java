package org.deeplearning4j.nn.activation;


import org.jblas.DoubleMatrix;

import static org.deeplearning4j.util.MatrixUtil.*;
import static org.jblas.MatrixFunctions.*;

public class Tanh extends BaseActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4499150153988528321L;

    /**
     * Name of the function
     *
     * @return the name of the function
     */
    @Override
    public String type() {
        return "tanh";
    }

    @Override
	public DoubleMatrix apply(DoubleMatrix arg0) {
		return tanh(arg0);
	}

	@Override
	public DoubleMatrix applyDerivative(DoubleMatrix input) {
		//1 - tanh^2 x
		return oneMinus(pow(tanh(input),2));
	}

	

}
