package org.nd4j.linalg.api.blas.impl;

import org.nd4j.linalg.api.blas.Level3;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Audrey Loeffel
 */
public class SparseBaseLevel3 extends SparseBaseLevel implements Level3 {
    @Override
    public void gemm(char Order, char TransA, char TransB, double alpha, INDArray A, INDArray B, double beta,
                    INDArray C) {

    }

    @Override
    public void gemm(INDArray A, INDArray B, INDArray C, boolean transposeA, boolean transposeB, double alpha,
                    double beta) {

    }

    @Override
    public void symm(char Order, char Side, char Uplo, double alpha, INDArray A, INDArray B, double beta, INDArray C) {

    }

    @Override
    public void syrk(char Order, char Uplo, char Trans, double alpha, INDArray A, double beta, INDArray C) {

    }

    @Override
    public void syr2k(char Order, char Uplo, char Trans, double alpha, INDArray A, INDArray B, double beta,
                    INDArray C) {

    }

    @Override
    public void trmm(char Order, char Side, char Uplo, char TransA, char Diag, double alpha, INDArray A, INDArray B,
                    INDArray C) {

    }

    @Override
    public void trsm(char Order, char Side, char Uplo, char TransA, char Diag, double alpha, INDArray A, INDArray B) {

    }

    @Override
    public void gemm(char Order, char TransA, char TransB, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray B,
                    IComplexNumber beta, IComplexNDArray C) {

    }

    @Override
    public void hemm(char Order, char Side, char Uplo, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray B,
                    IComplexNumber beta, IComplexNDArray C) {

    }

    @Override
    public void herk(char Order, char Uplo, char Trans, IComplexNumber alpha, IComplexNDArray A, IComplexNumber beta,
                    IComplexNDArray C) {

    }

    @Override
    public void her2k(char Order, char Uplo, char Trans, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray B,
                    IComplexNumber beta, IComplexNDArray C) {

    }

    @Override
    public void symm(char Order, char Side, char Uplo, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray B,
                    IComplexNumber beta, IComplexNDArray C) {

    }

    @Override
    public void syrk(char Order, char Uplo, char Trans, IComplexNumber alpha, IComplexNDArray A, IComplexNumber beta,
                    IComplexNDArray C) {

    }

    @Override
    public void syr2k(char Order, char Uplo, char Trans, IComplexNumber alpha, IComplexNDArray A, IComplexNDArray B,
                    IComplexNumber beta, IComplexNDArray C) {

    }

    @Override
    public void trmm(char Order, char Side, char Uplo, char TransA, char Diag, IComplexNumber alpha, IComplexNDArray A,
                    IComplexNDArray B, IComplexNDArray C) {

    }

    @Override
    public void trsm(char Order, char Side, char Uplo, char TransA, char Diag, IComplexNumber alpha, IComplexNDArray A,
                    IComplexNDArray B) {

    }
}
