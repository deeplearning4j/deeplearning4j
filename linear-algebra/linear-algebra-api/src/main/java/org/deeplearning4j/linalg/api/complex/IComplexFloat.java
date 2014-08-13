package org.deeplearning4j.linalg.api.complex;

/**
 * Complex float
 *
 * @author Adam Gibson
 */
public interface IComplexFloat extends IComplexNumber {
    @Override
    Float realComponent();

    @Override
    Float imaginaryComponent();
}
