package org.nd4j.linalg.ops.reduceops.complex;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Scalar ops for complex ndarrays
 *
 * @author Adam Gibson
 */
public class ComplexOps {


    public static IComplexNumber std(IComplexNDArray arr) {
        return new StandardDeviation().apply(arr);
    }
    public static IComplexNumber norm1(IComplexNDArray arr) {
        return new Norm1().apply(arr);
    }

    public static IComplexNumber norm2(IComplexNDArray arr) {
        return new Norm2().apply(arr);
    }
    public static IComplexNumber normmax(IComplexNDArray arr) {
        return new NormMax().apply(arr);
    }


    public static IComplexNumber max(IComplexNDArray arr) {
        return new Max().apply(arr);
    }

    public static IComplexNumber min(IComplexNDArray arr) {
        return new Min().apply(arr);
    }

    public static IComplexNumber mean(IComplexNDArray arr) {
        return new Mean().apply(arr);
    }

    public static IComplexNumber sum(IComplexNDArray arr) {
        return new Sum().apply(arr);
    }

    public static IComplexNumber var(IComplexNDArray arr) {
        return new Variance().apply(arr);
    }

    public static IComplexNumber prod(IComplexNDArray arr) {
        return new Prod().apply(arr);
    }
    
}
