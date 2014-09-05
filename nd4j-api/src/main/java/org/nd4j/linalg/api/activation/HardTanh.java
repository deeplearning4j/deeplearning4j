package org.nd4j.linalg.api.activation;


import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrays;
import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Tanh with a hard range of -1 < tanh(x) < 1
 * @author Adam Gibson
 */
public class HardTanh extends BaseActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8484119406683594852L;


    /**
     * The class used for transformation
     *
     * @return the class used for transformation
     */
    @Override
    public Class<? extends ElementWiseOp> transformClazz() {
        return org.nd4j.linalg.ops.transforms.HardTanh.class;
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
	public INDArray applyDerivative(INDArray input) {
		if(input instanceof IComplexNDArray) {
            for(int i = 0; i < input.length(); i++) {

                IComplexNumber val = (IComplexNumber) input.getScalar(i).element();
                if(val.realComponent().doubleValue() < -1 )
                     val.set(-1,val.imaginaryComponent().doubleValue());
                else if(val.realComponent().doubleValue() > 1)
                    val.set(1,val.imaginaryComponent().doubleValue());
                else
                    val = NDArrays.createDouble(1,0).subi(ComplexUtil.pow(ComplexUtil.tanh(val), 2));

                input.put(i, NDArrays.scalar(val));
            }
        }
        else {
            for(int i = 0; i < input.length(); i++) {

                double val = (double) input.getScalar(i).element();
                if(val < -1 )
                    val = -1;
                else if(val > 1)
                    val = 1;
                else
                    val = 1 - Math.pow(Math.tanh(val),2);
                input.put(i, NDArrays.scalar(val));
            }
        }

		
		return input;
		
	}

}
