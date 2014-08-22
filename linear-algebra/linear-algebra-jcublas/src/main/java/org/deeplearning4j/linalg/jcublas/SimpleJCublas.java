package org.deeplearning4j.linalg.jcublas;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.jcublas.complex.JCublasComplexNDArray;

/**
 * Created by mjk on 8/20/14.
 */
public class SimpleJCublas {
    public static JCublasNDArray gemv(JCublasNDArray A, JCublasNDArray B, double alpha, double beta) {

        JCublas.cublasInit();
        JCublas.setExceptionsEnabled(true);

        JCublasNDArray C = new JCublasNDArray(A.rows(), B.columns());

        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();

        JCublas.cublasAlloc(A.rows()*A.columns(), Sizeof.DOUBLE, d_A);
        JCublas.cublasAlloc(B.rows()*B.columns(), Sizeof.DOUBLE, d_B);
        JCublas.cublasAlloc(A.rows()*B.columns(), Sizeof.DOUBLE, d_C);

        JCublas.cublasSetVector(
                A.length(),
                Sizeof.DOUBLE,
                Pointer.to(A.data()),
                1,
                d_A,
                1);
        JCublas.cublasSetVector(
                B.length(),
                Sizeof.DOUBLE,
                Pointer.to(B.data()),
                1,
                d_B,
                1);

        JCublas.cublasDgemv(
                'n', //trans
                A.rows(),  // m
                B.columns(), // n
                alpha, //alpha
                d_A, // A
                A.rows(),  // lda
                d_B, // x
                1, // incx
                beta,  // beta
                d_C, // y
                1); // incy

        JCublas.cublasGetVector(
                C.length(),
                Sizeof.DOUBLE,
                d_C,
                1,
                Pointer.to(C.data),
                1);

        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);

        JCublas.cublasShutdown();

        return C;
    }
    public static JCublasComplexNDArray gemm(IComplexNDArray A_, IComplexNDArray B_, double alpha, double beta) {
    //cublasDnrm2(int n, jcuda.Pointer x, int incx)

        return null;
    }
    public static JCublasNDArray gemm(JCublasNDArray A_, JCublasNDArray B_, double alpha, double beta) {

        JCublas.cublasInit();
        JCublas.setExceptionsEnabled(true);
        /*
        JCublasNDArray A = new JCublasNDArray(A_.rows(), A_.columns(),A_.getOffsetData());
        JCublasNDArray B = new JCublasNDArray(B_.rows(),B_.columns(),B_.getOffsetData());
        */
        JCublasNDArray A = A_;
        JCublasNDArray B = B_;

        /*
        JCublasNDArray A = new JCublasNDArray(new int[]{A_.rows(),A_.columns()}, A_.stride(), A_.offset());
        JCublasNDArray B = new JCublasNDArray(new int[]{B_.rows(),B_.columns()}, B_.stride(), B_.offset());
        */
        JCublasNDArray C = new JCublasNDArray(A.rows(), B.columns());

        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();

        JCublas.cublasAlloc(A.rows()*A.columns(), Sizeof.DOUBLE, d_A);
        JCublas.cublasAlloc(B.rows()*B.columns(), Sizeof.DOUBLE, d_B);
        JCublas.cublasAlloc(A.rows()*B.columns(), Sizeof.DOUBLE, d_C);

        Pointer _a = new Pointer();
        Pointer _b = new Pointer();
        _a = _a.to(A.data);
        _b = _b.to(B.data);

        int ret;
        ret = JCublas.cublasSetMatrix(
                A.rows(),
                A.columns(),
                Sizeof.DOUBLE,
                _a,//.withByteOffset(A.offset()),
                A.rows(),
                d_A,
                A.rows()
        );
        ret = JCublas.cublasSetMatrix(
                B.rows(),
                B.columns(),
                Sizeof.DOUBLE,
                _b,//.withByteOffset(B.offset()),
                B.rows(),
                d_B,
                B.rows()
        );

        JCublas.cublasDgemm(
                'n', //trans
                'n',
                A.rows(),  // m
                B.columns(), // n
                B.rows(), //k,
                alpha,
                d_A, // A
                A.rows(),  // lda
                d_B, // x
                B.rows(), // incx
                beta,  // beta
                d_C, // y
                C.rows()); // incy

        ret = JCublas.cublasGetMatrix(
                C.rows(),
                C.columns(),
                Sizeof.DOUBLE,
                d_C,
                C.rows(),
                Pointer.to(C.data),
                C.rows());


        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);

        JCublas.cublasShutdown();

        return C;

        /*
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        int lda, ldb, ldc;
        lda = rows();
        ldb = columns();
        ldc = rows();
        JCublasNDArray temp = new JCublasNDArray(resultArray.shape(), ArrayUtil.calcStridesFortran(resultArray.shape()));


        JCublas.cublasAlloc(rows() * columns(), Sizeof.DOUBLE, d_A);
        JCublas.cublasAlloc(otherArray.rows()*otherArray.columns(), Sizeof.DOUBLE, d_B);
        JCublas.cublasAlloc(rows()*otherArray.columns(), Sizeof.DOUBLE, d_C);


        int ret;
        ret = JCublas.cublasSetMatrix(
                rows(),
                columns(),
                Sizeof.DOUBLE,
                Pointer.to(data),
                rows(),
                d_A,
                rows()
        );
        ret = JCublas.cublasSetMatrix(
                otherArray.rows(),
                otherArray.columns(),
                Sizeof.DOUBLE,
                Pointer.to(otherArray.data),
                otherArray.rows(),
                d_B,
                otherArray.rows()
        );
        JCublas.cublasDgemm(
                'n',
                'n',
                rows(),  // m
                otherArray.columns(),   // n
                columns(), // k
                1f,  // alpha
                d_A,
                lda,  // lda
                d_B,
                ldb, // ldb
                0.0f,  // beta
                d_C,
                ldc);  // ldc

        ret = JCublas.cublasGetMatrix(
                resultArray.rows(),
                resultArray.columns(),
                Sizeof.DOUBLE,
                d_C,
                rows(),
                Pointer.to(temp.data),
                rows());

        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);

        resultArray.copy(temp);
        return resultArray;
        */

    }

    public static void dcopy(int length, double[] data, int offset, int i, double[] data1, int i1, int i2) {
        Pointer X = new Pointer();
        Pointer Y = new Pointer();

        JCublas.cublasAlloc(length, Sizeof.DOUBLE, X);
        JCublas.cublasAlloc(length, Sizeof.DOUBLE, Y);

        JCublas.cublasInit();
        JCublas.setExceptionsEnabled(true);

        JCublas.cublasSetVector(
                length,
                Sizeof.DOUBLE,
                Pointer.to(data),
                1,
                X,
                1);

        JCublas.cublasDcopy(length,
                X,
                1,
                Y,
                1);

        JCublas.cublasGetVector(
                length,
                Sizeof.DOUBLE,
                Y,
                1,
                Pointer.to(data1),
                1);

        JCublas.cublasShutdown();

    }

    public static double nrm2(JCublasComplexNDArray jCublasComplexNDArray) {
        return 0;
    }
}
