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

package org.nd4j.linalg.api.blas.impl;

import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseSparseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;

/**
 * @author Audrey Loeffel
 */
public abstract class SparseBaseLevel1 extends SparseBaseLevel implements Level1 {

    /**
     * computes a vector-vector dot product.
     *
     * @param n number of accessed element
     * @param alpha
     * @param X an INDArray
     * @param Y an INDArray
     * @return the vector-vector dot product of X and Y
     */
    @Override
    public double dot(long n, double alpha, INDArray X, INDArray Y) {

        if (X instanceof BaseSparseNDArray) {
            BaseSparseNDArray sparseX = (BaseSparseNDArray) X;
            DataBuffer pointers = sparseX.getVectorCoordinates();

            switch (X.data().dataType()) {
                case DOUBLE:
                    DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, X, Y);
                    return ddoti(n, X, pointers, Y);
                case FLOAT:
                    DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, X, Y);
                    return sdoti(n, X, pointers, Y);
                case HALF:
                    DefaultOpExecutioner.validateDataType(DataBuffer.Type.HALF, X, Y);
                    return hdoti(n, X, pointers, Y);
                default:
            }
        }
        throw new UnsupportedOperationException();
    }


    @Override
    public double dot(long n, DataBuffer dx, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        throw new UnsupportedOperationException();
    }

    /**
     * Computes the Euclidean norm of a vector.
     *
     * @param arr a vector
     * @return the Euclidean norm of the vector
     */
    @Override
    public double nrm2(INDArray arr) {

        switch (arr.data().dataType()) {
            case DOUBLE:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, arr);
                return dnrm2(arr.length(), arr, 1);
            case FLOAT:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, arr);
                return snrm2(arr.length(), arr, 1);
            case HALF:
                return hnrm2(arr.length(), arr, 1);
            default:
        }
        throw new UnsupportedOperationException();
    }

    /**
     * Compute the sum of magnitude of the vector elements
     *
     * @param arr a vector
     * @return the sum of magnitude of the vector elements
     * */
    @Override
    public double asum(INDArray arr) {

        switch (arr.data().dataType()) {
            case DOUBLE:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, arr);
                return dasum(arr.length(), arr, 1);
            case FLOAT:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, arr);
                return sasum(arr.length(), arr, 1);
            case HALF:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.HALF, arr);
                return hasum(arr.length(), arr, 1);
            default:
        }
        throw new UnsupportedOperationException();
    }


    @Override
    public double asum(long n, DataBuffer x, int offsetX, int incrX) {
        throw new UnsupportedOperationException();
    }

    /**
     * Find the index of the element with maximum absolute value
     *
     * @param arr a vector
     * @return the index of the element with maximum absolute value
     * */
    @Override
    public int iamax(INDArray arr) {
        switch (arr.data().dataType()) {
            case DOUBLE:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, arr);
                return idamax(arr.length(), arr, 1);
            case FLOAT:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, arr);
                return isamax(arr.length(), arr, 1);
            case HALF:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.HALF, arr);
                return ihamax(arr.length(), arr, 1);
            default:
        }
        throw new UnsupportedOperationException();
    }

    @Override
    public int iamax(long n, INDArray arr, int stride) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int iamax(long n, DataBuffer x, int offsetX, int incrX) {
        throw new UnsupportedOperationException();
    }


    /**
     * Find the index of the element with maximum absolute value
     *
     * @param arr a vector
     * @return the index of the element with minimum absolute value
     * */
    @Override
    public int iamin(INDArray arr) {
        switch (arr.data().dataType()) {
            case DOUBLE:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, arr);
                return idamin(arr.length(), arr, 1);
            case FLOAT:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, arr);
                return isamin(arr.length(), arr, 1);
            case HALF:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.HALF, arr);
                return ihamin(arr.length(), arr, 1);
            default:
        }
        throw new UnsupportedOperationException();
    }

    @Override
    public void swap(INDArray x, INDArray y) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void copy(INDArray x, INDArray y) {
        // FIXME - for Raver119 :)
        throw new UnsupportedOperationException();
    }

    @Override
    public void copy(long n, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        throw new UnsupportedOperationException();
    }

    /**
     * Adds a scalar multiple of compressed sparse vector to a full-storage vector.
     *
     * @param n The number of element
     * @param alpha
     * @param x a sparse vector
     * @param y a dense vector
     *
     * */
    @Override
    public void axpy(long n, double alpha, INDArray x, INDArray y) {
        BaseSparseNDArray sparseX = (BaseSparseNDArray) x;
        DataBuffer pointers = sparseX.getVectorCoordinates();
        switch (x.data().dataType()) {
            case DOUBLE:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, x);
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.DOUBLE, y);
                daxpyi(n, alpha, x, pointers, y);
                break;
            case FLOAT:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, x);
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.FLOAT, y);
                saxpyi(n, alpha, x, pointers, y);
                break;
            case HALF:
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.HALF, x);
                DefaultOpExecutioner.validateDataType(DataBuffer.Type.HALF, y);
                haxpyi(n, alpha, x, pointers, y);
                break;
            default:
                throw new UnsupportedOperationException();
        }
    }

    @Override
    public void axpy(long n, double alpha, DataBuffer x, int offsetX, int incrX, DataBuffer y, int offsetY, int incrY) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void rotg(INDArray a, INDArray b, INDArray c, INDArray s) {
        throw new UnsupportedOperationException();
    }

    /**
     * Applies Givens rotation to sparse vectors one of which is in compressed form.
     *
     * @param N The number of elements in vectors X and Y
     * @param X a sparse vector
     * @param Y a full-storage vector
     * @param c a scalar
     * @param s a scalar
     *
     * */
    @Override
    public void rot(long N, INDArray X, INDArray Y, double c, double s) {


        if (X instanceof BaseSparseNDArray) {
            BaseSparseNDArray sparseX = (BaseSparseNDArray) X;

            switch (X.data().dataType()) {
                case DOUBLE:
                    droti(N, X, sparseX.getVectorCoordinates(), Y, c, s);
                    break;
                case FLOAT:
                    sroti(N, X, sparseX.getVectorCoordinates(), Y, c, s);
                    break;
                case HALF:
                    hroti(N, X, sparseX.getVectorCoordinates(), Y, c, s);
                    break;
                default:
                    throw new UnsupportedOperationException();
            }
        } else {
            throw new UnsupportedOperationException();
        }
    }

    @Override
    public void rotmg(INDArray d1, INDArray d2, INDArray b1, double b2, INDArray P) {
        throw new UnsupportedOperationException();
    }

    /**
     * Computes the product of a vector by a scalar.
     *
     * @param N The number of elements of the vector X
     * @param alpha a scalar
     * @param X a vector
     * */
    @Override
    public void scal(long N, double alpha, INDArray X) {
        switch (X.data().dataType()) {
            case DOUBLE:
                dscal(N, alpha, X, 1);
                break;
            case FLOAT:
                sscal(N, alpha, X, 1);
                break;
            case HALF:
                hscal(N, alpha, X, 1);
                break;
            default:
                throw new UnsupportedOperationException();
        }

    }

    @Override
    public boolean supportsDataBufferL1Ops() {
        return false;
    }


    /*
    * ===========================================================================
    * Prototypes for level 1 BLAS functions (complex are recast as routines)
    * ===========================================================================
    */

    protected abstract double ddoti(long N, INDArray X, DataBuffer indx, INDArray Y);

    protected abstract double sdoti(long N, INDArray X, DataBuffer indx, INDArray Y);

    protected abstract double hdoti(long N, INDArray X, DataBuffer indx, INDArray Y);

    protected abstract double snrm2(long N, INDArray X, int incx);

    protected abstract double dnrm2(long N, INDArray X, int incx);

    protected abstract double hnrm2(long N, INDArray X, int incx);

    protected abstract double dasum(long N, INDArray X, int incx);

    protected abstract double sasum(long N, INDArray X, int incx);

    protected abstract double hasum(long N, INDArray X, int incx);

    protected abstract int isamax(long N, INDArray X, int incx);

    protected abstract int idamax(long N, INDArray X, int incx);

    protected abstract int ihamax(long N, INDArray X, int incx);

    protected abstract int isamin(long N, INDArray X, int incx);

    protected abstract int idamin(long N, INDArray X, int incx);

    protected abstract int ihamin(long N, INDArray X, int incx);

    protected abstract void daxpyi(long N, double alpha, INDArray X, DataBuffer pointers, INDArray Y);

    protected abstract void saxpyi(long N, double alpha, INDArray X, DataBuffer pointers, INDArray Y);

    protected abstract void haxpyi(long N, double alpha, INDArray X, DataBuffer pointers, INDArray Y);

    protected abstract void droti(long N, INDArray X, DataBuffer indexes, INDArray Y, double c, double s);

    protected abstract void sroti(long N, INDArray X, DataBuffer indexes, INDArray Y, double c, double s);

    protected abstract void hroti(long N, INDArray X, DataBuffer indexes, INDArray Y, double c, double s);

    protected abstract void dscal(long N, double a, INDArray X, int incx);

    protected abstract void sscal(long N, double a, INDArray X, int incx);

    protected abstract void hscal(long N, double a, INDArray X, int incx);

}
