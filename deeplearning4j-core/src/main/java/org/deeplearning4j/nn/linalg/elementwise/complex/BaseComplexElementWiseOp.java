package org.deeplearning4j.nn.linalg.elementwise.complex;

import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.jblas.ComplexDouble;

/**
 * Base element wise operation for complex ndarrays
 *
 * @author Adam Gibson
 */
public abstract class BaseComplexElementWiseOp implements ComplexElementWiseOp {
    protected ComplexNDArray from,to;
    protected ComplexDouble scalarValue = new ComplexDouble(Double.NEGATIVE_INFINITY);


    protected BaseComplexElementWiseOp(ComplexNDArray from, ComplexNDArray to) {
        this.from = from.reshape(new int[]{1,from.length});
        this.to = to.reshape(new int[]{1,to.length});

        assert from.length == to.length : "From and to must be the same length";

    }

    protected BaseComplexElementWiseOp(ComplexNDArray from) {
        this.from = from;
    }

    protected BaseComplexElementWiseOp(ComplexNDArray from,ComplexDouble scalarValue) {
        this.from = from.reshape(new int[]{1,from.length});
        this.scalarValue = scalarValue;
    }

    /**
     * Apply the transformation at from[i]
     *
     * @param i the index of the element to applyTransformToOrigin
     */
    @Override
    public void applyTransformToOrigin(int i) {
        ComplexDouble get =  apply(getFromOrigin(i),i);
        from.put(i,get);

    }

    /**
     * Apply the transformation at from[i] using the supplied value
     *
     * @param i            the index of the element to applyTransformToOrigin
     * @param valueToApply the value to apply to the given index
     */
    @Override
    public void applyTransformToOrigin(int i, ComplexDouble valueToApply) {
        ComplexDouble ret = apply(valueToApply,i);
        from.put(i,ret);


    }

    @Override
    public ComplexDouble getFromOrigin(int i) {
        return from.get(i);
    }

    /**
     * The input matrix
     *
     * @return
     */
    @Override
    public ComplexNDArray from() {
        return from;
    }


}
