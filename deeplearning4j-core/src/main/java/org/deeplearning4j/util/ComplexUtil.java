package org.deeplearning4j.util;

import org.apache.commons.math3.util.FastMath;
import org.jblas.ComplexDouble;

/**
 * @author Adam Gibson
 */
public class ComplexUtil {

    public static ComplexDouble exp(ComplexDouble d) {
          return  new ComplexDouble(FastMath.exp(d.real()) * FastMath.cos(d.imag()),FastMath.exp(d.real()) * FastMath.sin(d.imag()));
    }


}
