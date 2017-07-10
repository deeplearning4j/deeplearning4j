/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.eigen;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Created by Luke Czapla on 7/9/17.
 */
public class Covariance {


    /**
     * Returns the covariance matrix of a dataset
     *
     * @param in A matrix of vectors of same length N, each as a row
     * @return an N x N covariance matrix
     */
    public static INDArray covarianceMatrix(INDArray in) {
        int dlength = in.size(0);
        int vlength = in.size(1);

        INDArray sum = Nd4j.create(vlength);
        INDArray product = Nd4j.create(vlength, vlength);

        for (int i = 0; i < vlength; i++)
            sum.getColumn(i).assign(in.getColumn(i).sumNumber().doubleValue()/dlength);

        for (int i = 0; i < dlength; i++) {
            INDArray dx1 = in.getRow(i).sub(sum);
            product.addi(dx1.reshape(vlength,1).mmul(dx1.reshape(1,vlength)));
        }
        product.divi(dlength);
        return product;
    }


    /**
     * Calculates the principal component vectors and their eigenvalues for the covariance matrix
     * @param cov The covariance matrix (calculated w/ covarianceMatrix(in) method)
     * @return An array of INDArray.  
     * 	The principal component vectors as column sorted in decreasing importance are element 0
     *      and the eigenvalues are element 1
     */
    public static INDArray[] principalComponents(INDArray cov) {
	assert cov.rows() == cov.columns();
        INDArray[] result = new INDArray[2];
        result[0] = Nd4j.eye(cov.rows());
        result[1] = Eigen.symmetricGeneralizedEigenvalues(result[0], cov, true);

        return result;
    }


}

