package org.nd4j.linalg.jcublas.complex;

import org.nd4j.linalg.api.complex.BaseComplexDouble;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Double implementation of a complex number.
 * Based on the jblas api by mikio braun
 *
 * @author Adam Gibson
 */
public class ComplexDouble extends BaseComplexDouble {

    public final static ComplexDouble UNIT = new ComplexDouble(1,0);
    public final static ComplexDouble NEG = new ComplexDouble(-1,0);
    public final static ComplexDouble ZERO = new ComplexDouble(0,0);

    public ComplexDouble(double real, double imag) {
        super(real, imag);
    }

    public ComplexDouble(double real) {
        super(real);
    }


    @Override
    public IComplexNumber dup() {
        return new ComplexDouble(real,imag);
    }

    /**
     * Convert to a float
     *
     * @return this complex number as a float
     */
    @Override
    public IComplexFloat asFloat() {
        return new ComplexFloat((float) real,(float) imag);
    }
}
