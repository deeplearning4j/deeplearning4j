package org.deeplearning4j.linalg.ops.transforms;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.BaseElementWiseOp;
import org.deeplearning4j.linalg.util.ComplexUtil;

/**
 * Power element wise op
 *
 * @author Adam Gibson
 */
public class Pow extends BaseElementWiseOp {


    private double power;
    private IComplexNumber powComplex;

    public Pow(Integer n) {
        this.power = n;
    }

    public Pow(Double n) {
        this.power = n;
    }

    public Pow(double power, IComplexNumber powComplex) {
        this.power = power;
        this.powComplex = powComplex;
    }

    public Pow(IComplexNumber powComplex) {
        this.powComplex = powComplex;
    }

    public Pow(double power) {
        this.power = power;
    }

    /**
     * The transformation for a given value (a scalar ndarray)
     *
     * @param value the value to applyTransformToOrigin (a scalar ndarray)
     * @param i     the index of the element being acted upon
     * @return the transformed value based on the input
     */
    @Override
    public INDArray apply(INDArray value, int i) {
        if(value instanceof IComplexNDArray) {
            IComplexNumber n = (IComplexNumber) value.element();
            return NDArrays.scalar(ComplexUtil.pow(n,power));
        }
        float d = (float) value.element();
        return NDArrays.scalar(Math.pow(d,power));
    }
}
