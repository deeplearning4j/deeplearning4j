/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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

    // FIXME: int cast !!!

    private SparseNd4jBlas sparseNd4jBlas = (SparseNd4jBlas) Nd4j.sparseFactory().blas();

    /**
     * Computes the dot product of a compressed sparse double vector by a full-storage real vector.
     * @param N The number of elements in x and indx
     * @param X an sparse INDArray. Size at least N
     * @param indx an Databuffer that Specifies the indices for the elements of x. Size at least N
     * @param Y a dense INDArray. Size at least max(indx[i])
     * */
    @Override
    protected double ddoti(long N, INDArray X, DataBuffer indx, INDArray Y) {
        return cblas_ddoti((int) N, (DoublePointer) X.data().addressPointer(),(IntPointer) indx.addressPointer(),
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
    protected double sdoti(long N, INDArray X, DataBuffer indx, INDArray Y) {
        return cblas_sdoti((int) N, (FloatPointer) X.data().addressPointer(),(IntPointer) indx.addressPointer(),
                (FloatPointer) Y.data().addressPointer());
    }

    @Override
    protected double hdoti(long N, INDArray X, DataBuffer indx, INDArray Y) {
        throw new UnsupportedOperationException();
    }

    /**
     * Computes the Euclidean norm of a float vector
     * @param N The number of elements in vector X
     * @param X an INDArray
     * @param incx the increment of X
     * */
    @Override
    protected double snrm2(long N, INDArray X, int incx){
        return cblas_snrm2((int) N, (FloatPointer) X.data().addressPointer(), incx);
    }

    /**
     * Computes the Euclidean norm of a double vector
     * @param N The number of elements in vector X
     * @param X an INDArray
     * @param incx the increment of X
     * */
    @Override
    protected double dnrm2(long N, INDArray X, int incx){
        return cblas_dnrm2((int) N, (DoublePointer) X.data().addressPointer(), incx);
    }

    @Override
    protected double hnrm2(long N, INDArray X, int incx){
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
    protected double dasum(long N, INDArray X, int incrx){
        return cblas_dasum((int) N, (DoublePointer) X.data().addressPointer(), incrx);
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
    protected double sasum(long N, INDArray X, int incrx){
        return cblas_sasum((int) N, (FloatPointer) X.data().addressPointer(), incrx);
    }

    @Override
    protected double hasum(long N, INDArray X, int incrx){
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
    protected int isamax(long N, INDArray X, int incX) {
        return (int) cblas_isamax((int) N, (FloatPointer) X.data().addressPointer(), incX);
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
    protected int idamax(long N, INDArray X, int incX) {
        return (int) cblas_idamax((int) N, (DoublePointer) X.data().addressPointer(), incX);
    }
    @Override
    protected int ihamax(long N, INDArray X, int incX) {
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
    protected int isamin(long N, INDArray X, int incX) {
        return (int) cblas_isamin((int) N, (FloatPointer) X.data().addressPointer(), incX);
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
    protected int idamin(long N, INDArray X, int incX) {
        return (int) cblas_idamin((int) N, (DoublePointer) X.data().addressPointer(), incX);
    }
    @Override
    protected int ihamin(long N, INDArray X, int incX) {
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
    protected void daxpyi(long N, double alpha, INDArray X, DataBuffer pointers, INDArray Y){
        cblas_daxpyi((int) N, alpha, (DoublePointer) X.data().addressPointer(), (IntPointer) pointers.addressPointer(),
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
    protected void saxpyi(long N, double alpha, INDArray X, DataBuffer pointers, INDArray Y) {
        cblas_saxpyi((int) N, (float) alpha, (FloatPointer) X.data().addressPointer(), (IntPointer) pointers.addressPointer(),
                (FloatPointer) Y.data().addressPointer());
    }

    @Override
    protected void haxpyi(long N, double alpha, INDArray X, DataBuffer pointers, INDArray Y){
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
    protected void droti(long N, INDArray X, DataBuffer indexes, INDArray Y, double c, double s) {
        cblas_droti((int) N, (DoublePointer) X.data().addressPointer(), (IntPointer) indexes.addressPointer(),
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
    protected void sroti(long N, INDArray X, DataBuffer indexes, INDArray Y, double c, double s) {
        cblas_sroti((int) N, (FloatPointer) X.data().addressPointer(), (IntPointer) indexes.addressPointer().capacity(X.columns()),
                (FloatPointer) Y.data().addressPointer(), (float) c, (float) s);
    }

    @Override
    protected void hroti(long N, INDArray X, DataBuffer indexes, INDArray Y, double c, double s) {
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
    protected void dscal(long N, double a, INDArray X, int incx) {
        cblas_dscal((int) N, a, (DoublePointer) X.data().addressPointer(), incx);
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
    protected void sscal(long N, double a, INDArray X, int incx) {
        cblas_sscal((int) N, (float) a, (FloatPointer) X.data().addressPointer(), incx);
    }

    @Override
    protected void hscal(long N, double a, INDArray X, int incx) {
        throw new UnsupportedOperationException();
    }
}
