package org.deeplearning4j.linalg.ops;


import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.util.Shape;

import java.util.concurrent.CountDownLatch;


/**
 * Apply an operation and save it to a resulting matrix
 *
 * @author Adam Gibson
 */
public abstract  class BaseTwoArrayElementWiseOp extends BaseElementWiseOp implements TwoArrayElementWiseOp {


    protected INDArray to,other;
    protected INDArray currTo,currOther;


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
        if(scalarValue == null) {
            if(currTo instanceof IComplexNDArray) {
                IComplexNumber number = (IComplexNumber) apply(destination, getOther(other,i), i);
                IComplexNDArray c2 = (IComplexNDArray) destination;
                c2.putScalar(i,number);
            }
            else {
                float f = (float)  apply(from, getOther(other,i), i);
                destination.putScalar(i,f);
            }

        }

        else {
            if(destination instanceof  IComplexNDArray) {
                IComplexNDArray c2 = (IComplexNDArray) destination;
                IComplexNumber n = (IComplexNumber) apply(destination,scalarValue,i);
                c2.putScalar(i,n);
            }

            float f = (float) apply(from,scalarValue,i);
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
                        applyTransformToOrigin(from,i);
                    else
                        applyTransformToOrigin(from,i,scalarValue);
        }
        else {

            assert from.length() == to.length() : "From and to must be same length";
            int num = from.vectorsAlongDimension(0);
            final CountDownLatch latch = new CountDownLatch(num);
            for(int i = 0; i < num; i++) {
                final int iDup = i;
                getThreads().execute(new Runnable() {
                    @Override
                    public void run() {
                        INDArray curr = to.vectorAlongDimension(iDup,0);
                        INDArray currOther = other != null ? other.vectorAlongDimension(iDup,0) : null;
                        INDArray fromCurr = from != null ? from.vectorAlongDimension(iDup,0) : null;



                        for(int j = 0; j < fromCurr.length(); j++) {
                            applyTransformToDestination(fromCurr,curr,currOther,j);
                        }

                        latch.countDown();
                    }
                });


            }

            try {
                latch.await();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
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
        if(currOther instanceof IComplexNDArray) {
            IComplexNDArray c = (IComplexNDArray) other;
            return c.getComplex(i);
        }

        return other.get(i);
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
                float element = (float) origin;
                return realComplex(element,otherValue);

            }


        }

        else {
            //complex + real
            if(origin instanceof IComplexNumber) {
                IComplexNumber firstValue = (IComplexNumber) origin;
                float realValue = (float) value;
                return complexReal(firstValue,realValue);

            }

            //both normal
            else {
                float firstElement = (float) origin;
                float secondElement = (float) value;
                return realReal(firstElement,secondElement);
            }


        }
    }


    protected abstract IComplexNumber complexComplex(IComplexNumber num1,IComplexNumber num2);

    protected abstract IComplexNumber realComplex(float real,IComplexNumber other);

    protected abstract IComplexNumber complexReal(IComplexNumber origin,float secondValue);

    protected abstract float realReal(float firstElement,float secondElement);

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
