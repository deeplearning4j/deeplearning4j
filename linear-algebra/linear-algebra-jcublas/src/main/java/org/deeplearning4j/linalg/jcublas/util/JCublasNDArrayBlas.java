package org.deeplearning4j.linalg.jcublas.util;

import org.deeplearning4j.linalg.jcublas.JCublasNDArray;
import org.deeplearning4j.linalg.jcublas.SimpleJCublas;
import org.deeplearning4j.linalg.jcublas.complex.JCublasComplexNDArray;

/**
 * Created by mjk on 8/20/14.
 */
public class JCublasNDArrayBlas {
    public static JCublasNDArray gemm(double alpha,
                                      JCublasNDArray A,
                                      JCublasNDArray B, double beta, JCublasNDArray C)
    {
        return SimpleJCublas.gemm(A,B,C, alpha, beta);
    }

    public static JCublasNDArray gemv(double alpha,
                                      JCublasNDArray A,
                                      JCublasNDArray B, double beta, JCublasNDArray C)
    {
        return SimpleJCublas.gemv(A,B,C, alpha, beta);
    }

    public static int nrm2(JCublasComplexNDArray x) {
        return SimpleJCublas.dznrm2(x.length(), x.data(), x.offset(), 1);
    }

    public static int asum(JCublasComplexNDArray x) {
        return SimpleJCublas.dzasum(x.length(), x.data(), x.offset(), 1);
    }

    public static int iamax(JCublasComplexNDArray x) {
        return SimpleJCublas.izamax(x.length(), x.data(), x.offset(), 1) - 1;    }
}
