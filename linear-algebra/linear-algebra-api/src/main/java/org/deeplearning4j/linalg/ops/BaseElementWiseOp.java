package org.deeplearning4j.linalg.ops;


import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


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
    protected static ExecutorService dimensionThreads;


    static {
        dimensionThreads = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() * 2);
        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            @Override
            public void run() {
                if(!getThreads().isShutdown())
                    getThreads().shutdown();
            }
        }));
    }

    public static  synchronized ExecutorService getThreads() {
        return dimensionThreads;
    }



    /**
     * Apply the transformation at from[i]
     *
     * @param i the index of the element to applyTransformToOrigin
     */
    @Override
    public void applyTransformToOrigin(INDArray origin,int i) {
        if(origin instanceof IComplexNumber) {
            IComplexNDArray c2 = (IComplexNDArray) origin;
            IComplexNumber transformed = apply(origin,getFromOrigin(origin,i),i);
            c2.putScalar(i,transformed);
        }
        else {
            float f =  apply(origin,getFromOrigin(origin,i),i);
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
                IComplexNumber apply =  apply(origin,valueToApply,i);
                c2.putScalar(i,apply);
            }
            else
                throw new IllegalArgumentException("Unable to apply a non complex number to a real ndarray");
        }
        else {
            float f = apply(origin,(float) valueToApply,i);
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
        final CountDownLatch latch = new CountDownLatch(from.vectorsAlongDimension(0));

        for(int i = 0; i < from.vectorsAlongDimension(0); i++) {
            final int dupI = i;
            getThreads().execute(new Runnable() {
                @Override
                public void run() {
                    INDArray vectorAlongDim = from.vectorAlongDimension(dupI,0);

                    for(int j = 0; j < vectorAlongDim.length(); j++) {
                        if(vectorAlongDim instanceof IComplexNDArray) {
                            IComplexNDArray c = (IComplexNDArray) vectorAlongDim;
                            IComplexNumber result =  apply(c,c.getComplex(j),j);
                            c.putScalar(j,result);
                        }
                        else {
                            float apply = apply(vectorAlongDim,vectorAlongDim.get(j),j);
                            vectorAlongDim.putScalar(j,apply);

                        }
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