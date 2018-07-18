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

package org.nd4j.linalg.inverse;

import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.CheckUtil;

/**
 * Created by agibsoncccc on 11/30/15.
 */
public class InvertMatrix {


    /**
     * Inverts a matrix
     * @param arr the array to invert
     * @param inPlace Whether to store the result in {@code arr}
     * @return the inverted matrix
     */
    public static INDArray invert(INDArray arr, boolean inPlace) {
        if (!arr.isSquare()) {
            throw new IllegalArgumentException("invalid array: must be square matrix");
        }

        //FIX ME: Please
       /* int[] IPIV = new int[arr.length() + 1];
        int LWORK = arr.length() * arr.length();
        INDArray WORK = Nd4j.create(new double[LWORK]);
        INDArray inverse = inPlace ? arr : arr.dup();
        Nd4j.getBlasWrapper().lapack().getrf(arr);
        Nd4j.getBlasWrapper().lapack().getri(arr.size(0),inverse,arr.size(0),IPIV,WORK,LWORK,0);*/

        RealMatrix rm = CheckUtil.convertToApacheMatrix(arr);
        RealMatrix rmInverse = new LUDecomposition(rm).getSolver().getInverse();


        INDArray inverse = CheckUtil.convertFromApacheMatrix(rmInverse);
        if (inPlace)
            arr.assign(inverse);
        return inverse;

    }

    /**
     * Calculates pseudo inverse of a matrix using QR decomposition
     * @param arr the array to invert
     * @return the pseudo inverted matrix
     */
    public static INDArray pinvert(INDArray arr, boolean inPlace) {

        // TODO : do it natively instead of relying on commons-maths

        RealMatrix realMatrix = CheckUtil.convertToApacheMatrix(arr);
        QRDecomposition decomposition = new QRDecomposition(realMatrix, 0);
        DecompositionSolver solver = decomposition.getSolver();

        if (!solver.isNonSingular()) {
            throw new IllegalArgumentException("invalid array: must be singular matrix");
        }

        RealMatrix pinvRM = solver.getInverse();

        INDArray pseudoInverse = CheckUtil.convertFromApacheMatrix(pinvRM);

        if (inPlace)
            arr.assign(pseudoInverse);
        return pseudoInverse;

    }

    /**
     * Compute the left pseudo inverse. Input matrix must have full column rank.
     *
     * See also: <a href="https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Definition">Moore–Penrose inverse</a>
     *
     * @param arr Input matrix
     * @param inPlace Whether to store the result in {@code arr}
     * @return Left pseudo inverse of {@code arr}
     * @exception IllegalArgumentException Input matrix {@code arr} did not have full column rank.
     */
    public static INDArray pLeftInvert(INDArray arr, boolean inPlace) {
        try {
          final INDArray inv = invert(arr.transpose().mmul(arr), inPlace).mmul(arr.transpose());
          if (inPlace) arr.assign(inv);
          return inv;
        } catch (SingularMatrixException e) {
          throw new IllegalArgumentException(
              "Full column rank condition for left pseudo inverse was not met.");
        }
    }

    /**
     * Compute the right pseudo inverse. Input matrix must have full row rank.
     *
     * See also: <a href="https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Definition">Moore–Penrose inverse</a>
     *
     * @param arr Input matrix
     * @param inPlace Whether to store the result in {@code arr}
     * @return Right pseudo inverse of {@code arr}
     * @exception IllegalArgumentException Input matrix {@code arr} did not have full row rank.
     */
    public static INDArray pRightInvert(INDArray arr, boolean inPlace) {
        try{
            final INDArray inv = arr.transpose().mmul(invert(arr.mmul(arr.transpose()), inPlace));
            if (inPlace) arr.assign(inv);
            return inv;
        } catch (SingularMatrixException e){
            throw new IllegalArgumentException(
                "Full row rank condition for right pseudo inverse was not met.");
        }
    }
}
