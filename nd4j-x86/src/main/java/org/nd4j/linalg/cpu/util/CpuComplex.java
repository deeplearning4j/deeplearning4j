package org.nd4j.linalg.cpu.util;

import org.jblas.ComplexDouble;
import org.jblas.ComplexFloat;

/**
 * Complex number conversions
 * between nd4j and jblas
 *
 * @author Adam Gibson
 */
public class CpuComplex {

    /**
     * Get a complex double from jblas
     * from an @link{IComplexDouble}
     * @param iComplexDouble
     * @return the jblas complex double from the given
     * icomplex number
     */
    public static ComplexDouble getComplexDouble(IComplexDouble iComplexDouble) {
        if(iComplexDouble instanceof org.nd4j.linalg.cpu.complex.ComplexDouble) {
            return (org.nd4j.linalg.cpu.complex.ComplexDouble) iComplexDouble;
        }
        else
            return new ComplexDouble(iComplexDouble.realComponent().doubleValue(),iComplexDouble.imaginaryComponent().doubleValue());
    }


    /**
     * Get a complex flloat from jblas
     * from an @link{IComplexFloat}
     * @param iComplexDouble
     * @return the jblas complex float from the given
     * icomplex number
     */
    public static ComplexFloat getComplexFloat(IComplexFloat iComplexDouble) {
        if(iComplexDouble instanceof org.nd4j.linalg.cpu.complex.ComplexDouble) {
            return (org.nd4j.linalg.cpu.complex.ComplexFloat) iComplexDouble;
        }
        else
            return new ComplexFloat(iComplexDouble.realComponent().floatValue(),iComplexDouble.imaginaryComponent().floatValue());
    }

}
