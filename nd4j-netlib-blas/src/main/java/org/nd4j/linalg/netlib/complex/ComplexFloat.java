package org.nd4j.linalg.netlib.complex;

import org.nd4j.linalg.api.complex.BaseComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Created by agibsoncccc on 6/11/15.
 */
public class ComplexFloat extends BaseComplexFloat {
    public ComplexFloat() {
    }

    public ComplexFloat(float real) {
        super(real);
    }

    public ComplexFloat(Float real, Float imag) {
        super(real, imag);
    }

    public ComplexFloat(float real, float imag) {
        super(real, imag);
    }

    @Override
    public IComplexNumber dup() {
        return new ComplexFloat(real,imag);
    }
}
