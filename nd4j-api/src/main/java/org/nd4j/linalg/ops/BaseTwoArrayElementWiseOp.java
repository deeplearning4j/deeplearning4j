package org.nd4j.linalg.ops;


import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.Shape;

import java.util.concurrent.CountDownLatch;


/**
 * Apply an operation and save it to a resulting matrix
 *
 * @author Adam Gibson
 */
public abstract  class BaseTwoArrayElementWiseOp extends BaseElementWiseOp implements TwoArrayElementWiseOp {


    protected INDArray to,other;


    /**
     * Apply the function from to the specified index
     * in to. The value from to is passed in to apply
     * and then a transform of the matching elements in
     * both from and to are used for a transformation.
     *
     * If a scalar is specified, this will apply a scalar wise operation
     * based on the scalar and the origin matrix instead
     * @param i the index to apply to
     */
    @Override
    public void applyTransformToDestination(INDArray from,INDArray destination,INDArray other,int i) {
        if(scalarValue != null) {
            if(destination instanceof IComplexNDArray) {
                IComplexNumber number = (IComplexNumber) apply(destination,  scalarValue, i);
                IComplexNDArray c2 = (IComplexNDArray) destination;
                c2.putScalar(i,number);
            }
            else {
                double f = (double)  apply(from,  scalarValue, i);
                destination.putScalar(i,f);
            }

        }

        else {
            if(destination instanceof  IComplexNDArray) {
                IComplexNDArray c2 = (IComplexNDArray) destination;
                IComplexNumber n = (IComplexNumber) apply(destination,getOther(other,i),i);
                c2.putScalar(i,n);
            }

            double f = (double) apply(from,getOther(other,i),i);
            destination.putScalar(i,f);

        }
    }

    /**
     * Executes the operation
     * across the matrix
     */
    @Override
    public void exec() {
        if(from != null && to != null && !from.isScalar() && !to.isScalar())
            assert Shape.shapeEquals(from.shape(),to.shape()) : "From and to must be same length";
        if(from != null && other != null && !from.isScalar() && !to.isScalar())
            assert from.length() == other.length() : "From and other must be the same length";

        if(to == null) {
            if(scalarValue != null)
                for(int i = 0; i < from.length(); i++)
                    if(scalarValue != null)
                        applyTransformToOrigin(from,i,scalarValue);
                    else
                        applyTransformToOrigin(from,i);
        }


        else if(other == null && scalarValue != null) {
            int num = from.vectorsAlongDimension(0);
            for(int i = 0; i < num; i++) {
                final int iDup = i;
                final  INDArray fromCurr = from != null ? from.vectorAlongDimension(iDup,0) : null;
                for(int j = 0; j < fromCurr.length(); j++) {
                    applyTransformToOrigin(fromCurr,j,scalarValue);
                }



            }


        }


        else {

            assert from.length() == to.length() : "From and to must be same length";
            int num = from.vectorsAlongDimension(0);
            for(int i = 0; i < num; i++) {
                final int iDup = i;
                final INDArray curr = to.vectorAlongDimension(iDup,0);
                final INDArray currOther = other != null ? other.vectorAlongDimension(iDup,0) : null;
                final  INDArray fromCurr = from != null ? from.vectorAlongDimension(iDup,0) : null;
                for(int j = 0; j < fromCurr.length(); j++) {
                    applyTransformToDestination(fromCurr,curr,currOther,j);
                }



            }


        }

    }

    /**
     * Returns the element
     * in destination at index i
     *
     * @param i the index of the element to retrieve
     * @return the element at index i
     */
    @Override
    public Object getOther(INDArray other, int i) {
        if(other instanceof IComplexNDArray) {
            IComplexNDArray c = (IComplexNDArray) other;
            return c.getComplex(i);
        }

        return other.getFloat(i);
    }


    protected Object doOp(INDArray originNDArray,int i,Object value) {
        Object origin = getFromOrigin(originNDArray,i);
        if(value instanceof IComplexNumber) {
            IComplexNDArray complexValue = (IComplexNDArray) value;
            IComplexNumber otherValue = (IComplexNumber) complexValue.element();
            //complex + complex
            if(origin instanceof IComplexNDArray) {
                IComplexNDArray originComplex = (IComplexNDArray) origin;
                IComplexNumber originValue = (IComplexNumber) originComplex.element();
                return complexComplex(originValue, otherValue);
            }

            //real + complex
            else {
                double element = (double) origin;
                return realComplex(element,otherValue);

            }


        }

        else {
            //complex + real
            if(origin instanceof IComplexNumber) {
                IComplexNumber firstValue = (IComplexNumber) origin;
                double realValue = (double) value;
                return complexReal(firstValue,realValue);

            }

            //both normal
            else {
                double firstElement = (double) origin;
                double secondElement = (double) value;
                return realReal(firstElement,secondElement);
            }


        }
    }


    protected abstract IComplexNumber complexComplex(IComplexNumber num1,IComplexNumber num2);

    protected abstract IComplexNumber realComplex(double real,IComplexNumber other);

    protected abstract IComplexNumber complexReal(IComplexNumber origin,double secondValue);

    protected abstract double realReal(double firstElement,double secondElement);

    /**
     * The transformation for a given value
     *
     * @param value the value to applyTransformToOrigin
     * @return the transformed value based on the input
     */
    @Override
    public Object apply(INDArray origin,Object value, int i) {
        return doOp(origin,i,value);
    }


    /**
     * The output matrix
     *
     * @return
     */
    @Override
    public INDArray to() {
        return to;
    }
}
