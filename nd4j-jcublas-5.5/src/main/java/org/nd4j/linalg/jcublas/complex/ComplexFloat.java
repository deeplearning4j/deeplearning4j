package org.nd4j.linalg.jcublas.complex;

import org.nd4j.linalg.api.complex.BaseComplexFloat;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Complex float
 * @author Adam Gibson
 */
public class ComplexFloat extends BaseComplexFloat  {


    public final static ComplexFloat UNIT = new ComplexFloat(1,0);
    public final static ComplexFloat NEG = new ComplexFloat(-1,0);
    public final static ComplexFloat ZERO = new ComplexFloat(0,0);

    public ComplexFloat(float real, float imag) {
        super(real, imag);
    }

    public ComplexFloat(float real) {
        super(real);
    }


    @Override
    public IComplexNumber dup() {
        return new ComplexFloat(real,imag);
    }
}
