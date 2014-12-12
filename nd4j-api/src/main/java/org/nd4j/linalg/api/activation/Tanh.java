package org.nd4j.linalg.api.activation;


import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.factory.ElementWiseOpFactories;
import org.nd4j.linalg.ops.factory.ElementWiseOpFactory;

/**
 * Tanh activation function
 *
 * @author Adam Gibson
 */
public class Tanh extends BaseActivationFunction {


    /**
     *
     */
    private static final long serialVersionUID = 4499150153988528321L;

    /**
     * The class used for transformation
     *
     * @return the class used for transformation
     */
    @Override
    public ElementWiseOpFactory transformFactory() {
        return ElementWiseOpFactories.tanh();
    }

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
    public INDArray applyDerivative(INDArray input) {
        //1 - tanh^2 x
        if(input instanceof IComplexNDArray) {
            return Nd4j.complexOnes(input.shape()).subi(input);
        }
        else
            return Nd4j.ones(input.shape()).subi(input);
    }




}
