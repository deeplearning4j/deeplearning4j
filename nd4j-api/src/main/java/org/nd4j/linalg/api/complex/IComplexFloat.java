package org.nd4j.linalg.api.complex;

/**
 * Complex float
 *
 * @author Adam Gibson
 */
public interface IComplexFloat extends IComplexNumber {
    IComplexFloat divi(float v);

    IComplexNumber div(float v);

    @Override
    Float complexArgument();

    @Override
    Float realComponent();

    @Override
    Float imaginaryComponent();

}
