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

    private PCA() {
    }

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


    /**
     * Calculates pca vectors of a matrix, for a given variance. A larger variance (99%)
     * will result in a higher order feature set.
     *
     *
     *
     * @param A the array of features, rows are results, columns are features
     * @param variance the amount of variance to preserve as a float 0 - 1
     * @return the matrix to mulitiply a feature by to get a reduced feature set
     */
    public static INDArray pca(INDArray A, double variance) {

        // Calculate the covariance matrix for the features
        INDArray cov = A.mmul( A.transpose() ) ;
        cov.divi( cov.meanNumber() );

        // Prepare SVD results, we'll decomp the covariance
        INDArray s = Nd4j.create( cov.rows() ) ;
        INDArray U = Nd4j.create( cov.rows(), cov.columns(), 'f' ) ;

        // Note - we don't care about VT (esp. since it will be the same as U for any AxA' )
        Nd4j.getBlasWrapper().lapack().sgesvd( cov, s, U, null );

        // Now find how many features we need to preserve the required variance
        // Which is the same percentage as a cumulative sum of the eigenvalues' percentages
        double totalEigSum = s.sumNumber().doubleValue() * variance ;
        int numEigsToKeep = -1 ;
        double runningTotal = 0 ;
        for( int i=0 ; i<s.length() ; i++ ) {
                runningTotal += s.getDouble(i) ;
                if( runningTotal >= totalEigSum ) {  // OK I know it's a float, but what else can we do ?
                        numEigsToKeep = i+1 ;	     // we will keep this many features to preserve the reqd. variance
                        break ;
                }
        }
	if( numEigsToKeep == -1 ) {   // if we need everything
		throw new RuntimeException( "No reduction possible for reqd. variance - use a smaller variance or don't use PCA" ) ;
	}
        // So now let's rip out the appropriate number of left singular vectors from
        // the U output
        INDArray factor = Nd4j.create( A.rows(), numEigsToKeep, 'f' ) ;
        for( int i=0 ; i<numEigsToKeep ; i++ ) {
                factor.putColumn( i, U.getColumn(i) ) ;
        }

        return factor  ;
    }

}
