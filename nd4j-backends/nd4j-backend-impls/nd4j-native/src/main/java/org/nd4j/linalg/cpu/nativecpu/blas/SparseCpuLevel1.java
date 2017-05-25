package org.nd4j.linalg.cpu.nativecpu.blas;

import org.nd4j.linalg.api.blas.impl.SparseBaseLevel1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.SparseNd4jBlas;
import org.bytedeco.javacpp.*;

import static org.bytedeco.javacpp.mkl_rt.*;

/**
 * @author Audrey Loeffel
 */
public class SparseCpuLevel1 extends SparseBaseLevel1 {

    private SparseNd4jBlas sparseNd4jBlas = (SparseNd4jBlas) Nd4j.sparseFactory().blas();

    /**
     * Computes the dot product of a compressed sparse double vector by a full-storage real vector.
     * @param N The number of elements in x and indx
     * @param X an sparse INDArray. Size at least N
     * @param indx an Databuffer that Specifies the indices for the elements of x. Size at least N
     * @param Y a dense INDArray. Size at least max(indx[i])
     * */
    @Override
    protected double ddoti(int N, INDArray X, DataBuffer indx, INDArray Y) {
        return cblas_ddoti(N, (DoublePointer) X.data().addressPointer(),(IntPointer) indx.addressPointer(),
                (DoublePointer) Y.data().addressPointer());
    }

    /**
     * Computes the dot product of a compressed sparse float vector by a full-storage real vector.
     * @param N The number of elements in x and indx
     * @param X an sparse INDArray. Size at least N
     * @param indx an Databuffer that specifies the indices for the elements of x. Size at least N
     * @param Y a dense INDArray. Size at least max(indx[i])
     * */
    @Override
    protected double sdoti(int N, INDArray X, DataBuffer indx, INDArray Y) {
        return cblas_sdoti(N, (FloatPointer) X.data().addressPointer(),(IntPointer) indx.addressPointer(),
                (FloatPointer) Y.data().addressPointer());
    }

    @Override
    protected double hdoti(int N, INDArray X, DataBuffer indx, INDArray Y) {
        throw new UnsupportedOperationException();
    }

    /**
     * Computes the Euclidean norm of a float vector
     * @param N The number of elements in vector X
     * @param X an INDArray
     * @param incx the increment of X
     * */
    @Override
    protected double snrm2(int N, INDArray X, int incx){
        return cblas_snrm2(N, (FloatPointer) X.data().addressPointer(), incx);
    }

    /**
     * Computes the Euclidean norm of a double vector
     * @param N The number of elements in vector X
     * @param X an INDArray
     * @param incx the increment of X
     * */
    @Override
    protected double dnrm2(int N, INDArray X, int incx){
        return cblas_dnrm2(N, (DoublePointer) X.data().addressPointer(), incx);
    }

    @Override
    protected double hnrm2(int N, INDArray X, int incx){
        throw new UnsupportedOperationException();
    }

    /**
     * Compute the sum of magnitude of the double vector elements
     *
     * @param N The number of elements in vector X
     * @param X a double vector
     * @param incrx The increment of X
     * @return the sum of magnitude of the vector elements
     * */
    @Override
    protected double dasum(int N, INDArray X, int incrx){
        return cblas_dasum(N, (DoublePointer) X.data().addressPointer(), incrx);
    }

    /**
     * Compute the sum of magnitude of the float vector elements
     *
     * @param N The number of elements in vector X
     * @param X a float vector
     * @param incrx The increment of X
     * @return the sum of magnitude of the vector elements
     * */
    @Override
    protected double sasum(int N, INDArray X, int incrx){
        return cblas_sasum(N, (FloatPointer) X.data().addressPointer(), incrx);
    }

    @Override
    protected double hasum(int N, INDArray X, int incrx){
        throw new UnsupportedOperationException();
    }

    /**
     * Find the index of the element with maximum absolute value
     *
     * @param N The number of elements in vector X
     * @param X a vector
     * @param incX The increment of X
     * @return the index of the element with maximum absolute value
     * */
    @Override
    protected int isamax(int N, INDArray X, int incX) {
        return (int) cblas_isamax(N, (FloatPointer) X.data().addressPointer(), incX);
    }
    /**
     * Find the index of the element with maximum absolute value
     *
     * @param N The number of elements in vector X
     * @param X a vector
     * @param incX The increment of X
     * @return the index of the element with maximum absolute value
     * */
    @Override
    protected int idamax(int N, INDArray X, int incX) {
        return (int) cblas_idamax(N, (DoublePointer) X.data().addressPointer(), incX);
    }
    @Override
    protected int ihamax(int N, INDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    /**
     * Find the index of the element with minimum absolute value
     *
     * @param N The number of elements in vector X
     * @param X a vector
     * @param incX The increment of X
     * @return the index of the element with minimum absolute value
     * */
    @Override
    protected int isamin(int N, INDArray X, int incX) {
        return (int) cblas_isamin(N, (FloatPointer) X.data().addressPointer(), incX);
    }

    /**
     * Find the index of the element with minimum absolute value
     *
     * @param N The number of elements in vector X
     * @param X a vector
     * @param incX The increment of X
     * @return the index of the element with minimum absolute value
     * */
    @Override
    protected int idamin(int N, INDArray X, int incX) {
        return (int) cblas_idamin(N, (DoublePointer) X.data().addressPointer(), incX);
    }
    @Override
    protected int ihamin(int N, INDArray X, int incX) {
        throw new UnsupportedOperationException();
    }

    /**
     * Adds a scalar multiple of double compressed sparse vector to a full-storage vector.
     *
     * @param N The number of elements in vector X
     * @param alpha
     * @param X a sparse vector
     * @param pointers A DataBuffer that specifies the indices for the elements of x.
     * @param Y a dense vector
     *
     * */
    @Override
    protected void daxpyi(int N, double alpha, INDArray X, DataBuffer pointers, INDArray Y){
        cblas_daxpyi(N, alpha, (DoublePointer) X.data().addressPointer(), (IntPointer) pointers.addressPointer(),
                (DoublePointer) Y.data().addressPointer());
    }

    /**
     * Adds a scalar multiple of float compressed sparse vector to a full-storage vector.
     *
     * @param N The number of elements in vector X
     * @param alpha
     * @param X a sparse vector
     * @param pointers A DataBuffer that specifies the indices for the elements of x.
     * @param Y a dense vector
     *
     * */
    @Override
    protected void saxpyi(int N, double alpha, INDArray X, DataBuffer pointers, INDArray Y) {
        cblas_saxpyi(N, (float) alpha, (FloatPointer) X.data().addressPointer(), (IntPointer) pointers.addressPointer(),
                (FloatPointer) Y.data().addressPointer());
    }

    @Override
    protected void haxpyi(int N, double alpha, INDArray X, DataBuffer pointers, INDArray Y){
        throw new UnsupportedOperationException();
    }

    /**
     * Applies Givens rotation to sparse vectors one of which is in compressed form.
     *
     * @param N The number of elements in vectors X and Y
     * @param X a double sparse vector
     * @param indexes The indexes of the sparse vector
     * @param Y a double full-storage vector
     * @param c a scalar
     * @param s a scalar
     * */
    @Override
    protected void droti(int N, INDArray X, DataBuffer indexes, INDArray Y, double c, double s) {
        cblas_droti(N, (DoublePointer) X.data().addressPointer(), (IntPointer) indexes.addressPointer(),
                (DoublePointer) Y.data().addressPointer(), c, s);
    }

    /**
     * Applies Givens rotation to sparse vectors one of which is in compressed form.
     *
     * @param N The number of elements in vectors X and Y
     * @param X a float sparse vector
     * @param indexes The indexes of the sparse vector
     * @param Y a float full-storage vector
     * @param c a scalar
     * @param s a scalar
     * */
    @Override
    protected void sroti(int N, INDArray X, DataBuffer indexes, INDArray Y, double c, double s) {
        cblas_sroti(N, (FloatPointer) X.data().addressPointer(), (IntPointer) indexes.addressPointer().capacity(X.columns()),
                (FloatPointer) Y.data().addressPointer(), (float) c, (float) s);
    }

    @Override
    protected void hroti(int N, INDArray X, DataBuffer indexes, INDArray Y, double c, double s) {
        throw new UnsupportedOperationException();
    }

    /**
     * Computes the product of a double vector by a scalar.
     *
     * @param N The number of elements of the vector X
     * @param a a scalar
     * @param X a vector
     * @param incx the increment of the vector X
     * */
    @Override
    protected void dscal(int N, double a, INDArray X, int incx) {
        cblas_dscal(N, a, (DoublePointer) X.data().addressPointer(), incx);
    }

    /**
     * Computes the product of a float vector by a scalar.
     *
     * @param N The number of elements of the vector X
     * @param a a scalar
     * @param X a vector
     * @param incx the increment of the vector X
     * */
    @Override
    protected void sscal(int N, double a, INDArray X, int incx) {
        cblas_sscal(N, (float) a, (FloatPointer) X.data().addressPointer(), incx);
    }

    @Override
    protected void hscal(int N, double a, INDArray X, int incx) {
        throw new UnsupportedOperationException();
    }
}
