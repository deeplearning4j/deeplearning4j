package org.deeplearning4j.linalg.jcublas.util;

import org.deeplearning4j.linalg.jcublas.JCublasNDArray;
import org.deeplearning4j.linalg.jcublas.SimpleJCublas;

/**
 * Created by mjk on 8/20/14.
 */
public class JCublasNDArrayBlas {
    public static JCublasNDArray gemm(double alpha,
                                      JCublasNDArray A,
                                      JCublasNDArray B, double beta, JCublasNDArray C)
    {
        return SimpleJCublas.gemm(A,B, alpha, beta);
    }

    public static JCublasNDArray gemv(double alpha,
                                      JCublasNDArray A,
                                      JCublasNDArray B, double beta, JCublasNDArray C)
    {
        return SimpleJCublas.gemv(A,B, alpha, beta);
    }
}
