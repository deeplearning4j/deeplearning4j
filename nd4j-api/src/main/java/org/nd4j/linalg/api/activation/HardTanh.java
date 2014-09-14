package org.nd4j.linalg.api.activation;


import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
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
            IComplexNDArray n2 = (IComplexNDArray) input;
            IComplexNDArray n2Linear = n2.linearView();
            for(int i = 0; i < n2Linear.length(); i++) {

                IComplexNumber val = n2Linear.getComplex(i);
                if(val.realComponent().doubleValue() < -1 )
                    val.set(-1,val.imaginaryComponent().doubleValue());
                else if(val.realComponent().doubleValue() > 1)
                    val.set(1,val.imaginaryComponent().doubleValue());
                else
                    val = Nd4j.createDouble(1, 0).subi(ComplexUtil.pow(ComplexUtil.tanh(val), 2));

                n2Linear.putScalar(i, val);
            }
        }
        else {
            INDArray linear = input.linearView();

            for(int i = 0; i < linear.length(); i++) {

                float val = linear.get(i);
                if(val < -1 )
                    val = -1;
                else if(val > 1)
                    val = 1;
                else
                    val = 1 - (float) Math.pow(Math.tanh(val),2);
                linear.putScalar(i,val);
            }
        }


        return input;

    }

}
