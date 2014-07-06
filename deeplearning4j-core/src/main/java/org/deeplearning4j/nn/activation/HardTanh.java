package org.deeplearning4j.nn.activation;

import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

/**
 * Tanh with a hard range of -1,t
 */
public class HardTanh extends BaseActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8484119406683594852L;


    @Override
    public FloatMatrix apply(FloatMatrix matrix) {
        for(int i = 0; i < matrix.length; i++) {
            double val = matrix.get(i);
            if(val < -1 )
                val = -1;
            else if(val > 1)
                val = 1;
            else
                val = Math.tanh(val);
            matrix.put(i,(float) val);
        }

        return matrix;
    }

    @Override
    public FloatMatrix applyDerivative(FloatMatrix input) {
        for(int i = 0; i < input.length; i++) {
            double val = input.get(i);
            if(val < -1 )
                val = -1;
            else if(val > 1)
                val = 1;
            else
                val = 1 - Math.pow(Math.tanh(val),2);
            input.put(i,(float) val);
        }

        return input;
    }


    /**
     * Name of the function
     *
     * @return the name of the function
     */
    @Override
    public String type() {
        return "hardtanh";
    }

    @Override
	public DoubleMatrix apply(DoubleMatrix matrix) {
		for(int i = 0; i < matrix.length; i++) {
			double val = matrix.get(i);
			if(val < -1 )
				val = -1;
			else if(val > 1)
				val = 1;
			else
				val = Math.tanh(val);
			matrix.put(i,val);
		}
		
		return matrix;
	}

	@Override
	public DoubleMatrix applyDerivative(DoubleMatrix input) {
		for(int i = 0; i < input.length; i++) {
			double val = input.get(i);
			if(val < -1 )
				val = -1;
			else if(val > 1)
				val = 1;
			else
				val = 1 - Math.pow(Math.tanh(val),2);
			input.put(i,val);
		}
		
		return input;
		
	}

}
