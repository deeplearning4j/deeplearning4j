package org.deeplearning4j.util;

import org.apache.commons.math3.util.FastMath;
import org.jblas.ComplexDouble;

/**
 * @author Adam Gibson
 */
public class ComplexUtil {

    /**
     * Returns the exp of a complex number:
     * Let r be the real component and i be the imaginary
     * Let ret be the complex number returned
     * ret -> exp(r) * cos(i), exp(r) * sin(i)
     * where the first number is the real component
     * and the second number is the imaginary component
     * @param d the number to get the exp of
     * @return the exponential of this complex number
     */
    public static ComplexDouble exp(ComplexDouble d) {
          return  new ComplexDouble(FastMath.exp(d.real()) * FastMath.cos(d.imag()),FastMath.exp(d.real()) * FastMath.sin(d.imag()));
    }


}
