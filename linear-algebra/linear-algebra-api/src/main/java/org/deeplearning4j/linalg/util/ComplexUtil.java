package org.deeplearning4j.linalg.util;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.linalg.api.complex.IComplexDouble;
import org.deeplearning4j.linalg.api.complex.IComplexFloat;
import org.deeplearning4j.linalg.factory.NDArrays;

/**
 * @author Adam Gibson
 */
public class ComplexUtil {

    /**
     * Returns the exp of a complex number:
     * Let r be the realComponent component and i be the imaginary
     * Let ret be the complex number returned
     * ret -> exp(r) * cos(i), exp(r) * sin(i)
     * where the first number is the realComponent component
     * and the second number is the imaginary component
     * @param d the number to getFromOrigin the exp of
     * @return the exponential of this complex number
     */
    public static IComplexDouble exp(IComplexDouble d) {
          return  NDArrays.createDouble(FastMath.exp(d.realComponent()) * FastMath.cos(d.imaginaryComponent()), FastMath.exp(d.realComponent()) * FastMath.sin(d.imaginaryComponent()));
    }

    /**
     * Returns the exp of a complex number:
     * Let r be the realComponent component and i be the imaginary
     * Let ret be the complex number returned
     * ret -> exp(r) * cos(i), exp(r) * sin(i)
     * where the first number is the realComponent component
     * and the second number is the imaginary component
     * @param d the number to getFromOrigin the exp of
     * @return the exponential of this complex number
     */
    public static IComplexFloat exp(IComplexFloat d) {
        return  NDArrays.createFloat((float) FastMath.exp(d.realComponent()) * (float) FastMath.cos(d.imaginaryComponent()),(float) FastMath.exp(d.realComponent()) * (float) FastMath.sin(d.imaginaryComponent()));
    }



}
