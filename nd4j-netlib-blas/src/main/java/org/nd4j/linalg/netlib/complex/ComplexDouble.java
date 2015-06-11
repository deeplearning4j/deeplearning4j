package org.nd4j.linalg.netlib.complex;

import org.nd4j.linalg.api.complex.BaseComplexDouble;

/**
 * Created by agibsoncccc on 6/11/15.
 */
public class ComplexDouble extends BaseComplexDouble {
    public ComplexDouble() {
    }

    public ComplexDouble(double real) {
        super(real);
    }

    public ComplexDouble(Double real, Double imag) {
        super(real, imag);
    }

    public ComplexDouble(double real, double imag) {
        super(real, imag);
    }
}
