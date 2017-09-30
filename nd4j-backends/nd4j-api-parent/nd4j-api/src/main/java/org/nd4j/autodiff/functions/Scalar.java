package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.samediff.SameDiff;


/**
 * Scalar value
 *
 */
public class Scalar extends Constant {

    protected double value;

    public Scalar(SameDiff sameDiff,
                  double value) {
        this(sameDiff, value, false);

    }

    public Scalar(SameDiff sameDiff,
                  double value,boolean inPlace) {
        super(sameDiff,  sameDiff.getArrayFactory().scalar(value),new int[]{1,1},inPlace);
        this.value = value;

    }



    @Override
    public DifferentialFunction dup() {
        return new Scalar(sameDiff, value);
    }
}
