/*
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

package org.nd4j.linalg.dimensionalityreduction;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * PCA class for dimensionality reduction
 *
 * @author Adam Gibson
 */
public class PCA {

    /**
     * Reduce the dimension of x
     * to the specified number of dimensions.
     * <p/>
     * Happily based on the great work done in the tsne paper here:
     * http://homepage.tudelft.nl/19j49/t-SNE.html
     *
     * @param X         the x to reduce
     * @param nDims     the number of dimensions to reduce to
     * @param normalize normalize
     * @return the reduced dimension
     */
    public static INDArray pca(INDArray X, int nDims, boolean normalize) {
        if (normalize) {
            INDArray mean = X.mean(0);
            X = X.subiRowVector(mean);

        }

        INDArray C;
        if (X.size(1) < X.size(0))
            C = X.transpose().mmul(X);

        else
            C = X.mmul(X.transpose()).muli(1 / X.size(0));

        IComplexNDArray[] eigen = Eigen.eigenvectors(C);

        IComplexNDArray M = eigen[1];
        IComplexNDArray lambda = eigen[0];
        IComplexNDArray diagLambda = Nd4j.diag(lambda);
        INDArray[] sorted = Nd4j.sortWithIndices(diagLambda, 0, false);
        //change lambda to be the indexes


        INDArray indices = sorted[0];

        INDArrayIndex[] indices2 = NDArrayIndex.create(indices.get(
                NDArrayIndex.interval(0, nDims)));

        INDArrayIndex[] rowsAndColumnIndices = new INDArrayIndex[]{
                NDArrayIndex.interval(0, M.rows()), indices2[0]
        };

        M = M.get(rowsAndColumnIndices);

        X = Nd4j.createComplex(X.subRowVector(X.mean(0))).mmul(M);


        return X;


    }


}
