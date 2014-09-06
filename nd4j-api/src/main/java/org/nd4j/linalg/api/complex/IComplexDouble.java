package org.nd4j.linalg.api.complex;

/**
 * Complex Double
 *
 * @author Adam Gibson
 */
public interface IComplexDouble extends IComplexNumber {
    @Override
    Double realComponent();

    @Override
    Double imaginaryComponent();

    IComplexDouble divi(double v);

    IComplexNumber div(double v);


}
