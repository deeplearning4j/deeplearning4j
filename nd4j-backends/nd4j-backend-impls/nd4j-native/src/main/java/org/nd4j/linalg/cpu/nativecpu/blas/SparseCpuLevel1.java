package org.nd4j.linalg.cpu.nativecpu.blas;

import org.nd4j.linalg.api.blas.impl.SparseBaseLevel1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.iter.FlatIterator;
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
        return cblas_ddoti(N, (DoublePointer) X.data().addressPointer(),(IntPointer) indx.addressPointer(), (DoublePointer) Y.data().addressPointer());
    }

    /**
     * Computes the dot product of a compressed sparse float vector by a full-storage real vector.
     * @param N The number of elements in x and indx
     * @param X an sparse INDArray. Size at least N
     * @param indx an Databuffer that Specifies the indices for the elements of x. Size at least N
     * @param Y a dense INDArray. Size at least max(indx[i])
     * */
    @Override
    protected double sdoti(int N, INDArray X, DataBuffer indx, INDArray Y) {
        return cblas_sdoti(N, (FloatPointer) X.data().addressPointer(),(IntPointer) indx.addressPointer(), (FloatPointer) Y.data().addressPointer());
    }

    @Override
    protected double hdoti(int N, INDArray X, DataBuffer indx, INDArray Y) {
        throw new UnsupportedOperationException();
    }

    /**
     * Computes the Euclidean norm of a float vector
     * @param N The number of element in vector X
     * @param X an INDArray
     * @param incx the increment of X
     * */
    @Override
    protected double snrm2(int N, INDArray X, int incx){
        return cblas_snrm2(N, (FloatPointer) X.data().addressPointer(), incx);
    }

    /**
     * Computes the Euclidean norm of a double vector
     * @param N The number of element in vector X
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
     * @param N The number of element in vector X
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
     * @param N The number of element in vector X
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
     * @param N The number of element in vector X
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
     * @param N The number of element in vector X
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
     * @param N The number of element in vector X
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
     * @param N The number of element in vector X
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
     * Swap a vector with another vector
     *
     * @param N The number of element in vector X and Y
     * @param X a vector
     * @param incrx The increment of X
     * @param Y a vector
     * @param incry The increment of Y
     * */
   /* @Override
    protected void dswap(int N, INDArray X, int incrx, INDArray Y, int incry){
        cblas_dswap(N, (DoublePointer) X.data().addressPointer(), incrx, (DoublePointer) Y.data().addressPointer(), incry);
    }
    */

    /**
     * Swap a vector with another vector
     *
     * @param N The number of element in vector X and Y
     * @param X a vector
     * @param incrx The increment of X
     * @param Y a vector
     * @param incry The increment of Y
     * */
    /*
    @Override
    protected void sswap(int N, INDArray X, int incrx, INDArray Y, int incry){
        cblas_sswap(N, (FloatPointer) X.data().addressPointer(), incrx, (FloatPointer) Y.data().addressPointer(), incry);

    }

    @Override
    protected void hswap(int N, INDArray X, int incrx, INDArray Y, int incry){
        throw new UnsupportedOperationException();
    }
    /*
    /**
     * Swap a vector with another vector
     *
     * @param N The number of element in vector X and Y
     * @param X a vector
     * @param incrx The increment of X
     * @param Y a vector
     * @param incry The increment of Y
     * */

    @Override
    protected int scopy(int N, INDArray X, int incrx, INDArray Y, int incry) {
        return 0;
    }

    @Override
    protected int dcopy(int N, INDArray X, int incrx, INDArray Y, int incry) {
        return 0;
    }

    @Override
    protected int hcopy(int N, INDArray X, int incrx, INDArray Y, int incry) {
        return 0;
    }
}
