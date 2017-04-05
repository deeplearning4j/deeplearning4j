package org.nd4j.linalg.cpu.nativecpu.blas;

import org.nd4j.linalg.api.blas.impl.SparseBaseLevel1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.IntBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.SparseNd4jBlas;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;

import java.nio.FloatBuffer;

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
    protected double ddoti(int N, INDArray X, IntBuffer indx, INDArray Y) {

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
    protected double sdoti(int N, INDArray X, IntBuffer indx, INDArray Y) {

        return cblas_sdoti(N, (FloatPointer) X.data().addressPointer(),(IntPointer) indx.addressPointer(), (FloatPointer) Y.data().addressPointer());

    }

}
