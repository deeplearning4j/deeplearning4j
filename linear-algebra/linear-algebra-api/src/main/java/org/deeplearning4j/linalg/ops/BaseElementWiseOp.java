package org.deeplearning4j.linalg.ops;


import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;


/**
 * Baseline element wise operation so only applyTransformToOrigin has to be implemented.
 * This also handles the ability to perform scalar wise operations vs just
 * a functional transformation
 *
 * @author Adam Gibson
 */

public abstract class BaseElementWiseOp implements ElementWiseOp {

    protected INDArray from;
    //this is for operations like adding or multiplying a scalar over the from array
    protected Object scalarValue;
    protected INDArray currVector;


    /**
     * Apply the transformation at from[i]
     *
     * @param i the index of the element to applyTransformToOrigin
     */
    @Override
    public void applyTransformToOrigin(INDArray origin,int i) {
        if(origin instanceof IComplexNumber) {
            IComplexNDArray c2 = (IComplexNDArray) origin;
            IComplexNumber transformed = (IComplexNumber) apply(origin,getFromOrigin(origin,i),i);
            c2.putScalar(i,transformed);
        }
        else {
            float f = (float) apply(origin,getFromOrigin(origin,i),i);
            origin.putScalar(i, f);
        }

    }

    /**
     * Apply the transformation at from[i] using the supplied value
     * @param origin the origin ndarray
     *  @param i            the index of the element to applyTransformToOrigin
     * @param valueToApply the value to apply to the given index
     */
    @Override
    public void applyTransformToOrigin(INDArray origin,int i, Object valueToApply) {
        if(valueToApply instanceof IComplexNumber) {
            if(origin instanceof IComplexNDArray) {
                IComplexNDArray c2 = (IComplexNDArray) origin;
                IComplexNumber apply = (IComplexNumber) apply(origin,valueToApply,i);
                c2.putScalar(i,apply);
            }
            else
                throw new IllegalArgumentException("Unable to apply a non complex number to a real ndarray");
        }
        else {
            float f = (float) apply(origin,valueToApply,i);
            origin.putScalar(i,f);
        }



    }

    @Override
    public Object getFromOrigin(INDArray origin,int i) {
        if(origin instanceof IComplexNDArray) {
            IComplexNDArray c2 = (IComplexNDArray) origin;
            return c2.getComplex(i);
        }

        return origin.get(i);
    }

    /**
     * The input matrix
     *
     * @return
     */
    @Override
    public INDArray from() {
        return from;
    }

    /**
     * Apply the transformation
     */
    @Override
    public void exec() {
        for(int i = 0; i < from.vectorsAlongDimension(0); i++) {

            INDArray vectorAlongDim = from.vectorAlongDimension(i,0);
            currVector = vectorAlongDim;

            for(int j = 0; j < vectorAlongDim.length(); j++) {
                if(vectorAlongDim instanceof IComplexNDArray) {
                    IComplexNDArray c = (IComplexNDArray) vectorAlongDim;
                    IComplexNumber result = (IComplexNumber)  apply(c,c.getComplex(j),j);
                    c.putScalar(i,result);
                }
                else {
                    Object apply = apply(vectorAlongDim,vectorAlongDim.get(j),j);
                    float f = (float) apply ;
                    vectorAlongDim.putScalar(j,f);

                }
            }
        }




    }
}
