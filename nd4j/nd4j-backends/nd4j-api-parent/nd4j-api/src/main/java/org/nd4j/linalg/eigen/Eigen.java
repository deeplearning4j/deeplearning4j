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

package org.nd4j.linalg.eigen;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;

/**
 * Compute eigen values
 *
 * @author Adam Gibson
 */
public class Eigen {

    public static INDArray dummy = Nd4j.scalar(1);

    /**
     * Computes the eigenvalues of a general matrix.
     */
    public static IComplexNDArray eigenvalues(INDArray A) {
        assert A.rows() == A.columns();
        INDArray WR = Nd4j.create(A.rows(), A.rows());
        INDArray WI = WR.dup();
        Nd4j.getBlasWrapper().geev('N', 'N', A.dup(), WR, WI, dummy, dummy);
        return Nd4j.createComplex(WR, WI);
    }

    /**
     * Compute generalized eigenvalues of the problem A x = L x.
     * Matrix A is modified in the process, holding eigenvectors after execution.
     *
     * @param A symmetric Matrix A. After execution, A will contain the eigenvectors as columns
     * @return a vector of eigenvalues L.
     */
    public static INDArray symmetricGeneralizedEigenvalues(INDArray A) {
        INDArray eigenvalues = Nd4j.create(A.rows());
        Nd4j.getBlasWrapper().syev('V', 'L', A, eigenvalues);
        return eigenvalues;
    }


    /**
     * Compute generalized eigenvalues of the problem A x = L x.
     * Matrix A is modified in the process, holding eigenvectors as columns after execution.
     *
     * @param A symmetric Matrix A. After execution, A will contain the eigenvectors as columns
     * @param calculateVectors if false, it will not modify A and calculate eigenvectors
     * @return a vector of eigenvalues L.
     */
    public static INDArray symmetricGeneralizedEigenvalues(INDArray A, boolean calculateVectors) {
        INDArray eigenvalues = Nd4j.create(A.rows());
        Nd4j.getBlasWrapper().syev('V', 'L', (calculateVectors ? A : A.dup()), eigenvalues);
        return eigenvalues;
    }


    /**
     * Computes the eigenvalues and eigenvectors of a general matrix.
     * <p/>
     * For matlab users note the following from their documentation:
     * The columns of V present eigenvectors of A. The diagonal matrix D contains eigenvalues.
     * <p/>
     * This is in reverse order of the matlab eig(A) call.
     *
     * @param A the ndarray to getFloat the eigen vectors for
     * @return 2 arrays representing W (eigen vectors) and V (normalized eigen vectors)
     */
    public static IComplexNDArray[] eigenvectors(INDArray A) {
        assert A.columns() == A.rows();
        // setting up result arrays
        INDArray WR = Nd4j.create(A.rows());
        INDArray WI = WR.dup();
        INDArray VR = Nd4j.create(A.rows(), A.rows());
        INDArray VL = Nd4j.create(A.rows(), A.rows());

        Nd4j.getBlasWrapper().geev('v', 'v', A.dup(), WR, WI, VL, VR);

        // transferring the result
        IComplexNDArray E = Nd4j.createComplex(WR, WI);
        IComplexNDArray V = Nd4j.createComplex((int) A.rows(), (int) A.rows());
        for (int i = 0; i < A.rows(); i++) {
            if (E.getComplex(i).isReal()) {
                IComplexNDArray column = Nd4j.createComplex(VR.getColumn(i));
                V.putColumn(i, column);
            } else {
                IComplexNDArray v = Nd4j.createComplex(VR.getColumn(i), VR.getColumn(i + 1));
                V.putColumn(i, v);
                V.putColumn(i + 1, v.conji());
                i += 1;
            }
        }
        return new IComplexNDArray[] {Nd4j.diag(E), V};
    }


    /**
     * Compute generalized eigenvalues of the problem A x = L B x.
     * The data will be unchanged, no eigenvectors returned.
     *
     * @param A symmetric Matrix A.
     * @param B symmetric Matrix B.
     * @return a vector of eigenvalues L.
     */
    public static INDArray symmetricGeneralizedEigenvalues(INDArray A, INDArray B) {
        assert A.rows() == A.columns();
        assert B.rows() == B.columns();
        INDArray W = Nd4j.create(A.rows());

        A = InvertMatrix.invert(B, false).mmuli(A);
        Nd4j.getBlasWrapper().syev('V', 'L', A, W);
        return W;
    }

    /**
     * Compute generalized eigenvalues of the problem A x = L B x.
     * The data will be unchanged, no eigenvectors returned unless calculateVectors is true.
     * If calculateVectors == true, A will contain a matrix with the eigenvectors as columns.
     *
     * @param A symmetric Matrix A.
     * @param B symmetric Matrix B.
     * @return a vector of eigenvalues L.
     */
    public static INDArray symmetricGeneralizedEigenvalues(INDArray A, INDArray B, boolean calculateVectors) {
        assert A.rows() == A.columns();
        assert B.rows() == B.columns();
        INDArray W = Nd4j.create(A.rows());
        if (calculateVectors)
            A.assign(InvertMatrix.invert(B, false).mmuli(A));
        else
            A = InvertMatrix.invert(B, false).mmuli(A);

        Nd4j.getBlasWrapper().syev('V', 'L', A, W);
        return W;
    }


}
