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

package org.nd4j.linalg.dimensionalityreduction;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * PCA class for dimensionality reduction
 *
 * @author Adam Gibson
 */
public class PCA {

    private PCA() {}


    /**
     * Calculates pca vectors of a matrix, for a fixed number of reduced features
     * returns the reduced feature set
     * The return is a projection of A onto principal nDims components
     *
     * To use the PCA: assume A is the original feature set
     * then project A onto a reduced set of features. It is possible to 
     * reconstruct the original data ( losing information, but having the same
     * dimensionality )
     *
     * <pre>
     * {@code
     *
     * INDArray Areduced = A.mmul( factor ) ;
     * INDArray Aoriginal = Areduced.mmul( factor.transpose() ) ;
     * 
     * }
     * </pre>
     *
     * @param A the array of features, rows are results, columns are features - will be changed
     * @param nDims the number of components on which to project the features 
     * @param normalize whether to normalize (adjust each feature to have zero mean)
     * @return the reduced parameters of A
     */
    public static INDArray pca(INDArray A, int nDims, boolean normalize) {
        INDArray factor = pca_factor(A, nDims, normalize);
        return A.mmul(factor);
    }



    /**
     * Calculates pca factors of a matrix, for a fixed number of reduced features
     * returns the factors to scale observations 
     *
     * The return is a factor matrix to reduce (normalized) feature sets
     *
     * @see pca(INDArray, int, boolean)
     *
     * @param A the array of features, rows are results, columns are features - will be changed
     * @param nDims the number of components on which to project the features 
     * @param normalize whether to normalize (adjust each feature to have zero mean)
     * @return the reduced feature set
     */
    public static INDArray pca_factor(INDArray A, int nDims, boolean normalize) {

        if (normalize) {
            // Normalize to mean 0 for each feature ( each column has 0 mean )
            INDArray mean = A.mean(0);
            A.subiRowVector(mean);
        }

        int m = A.rows();
        int n = A.columns();

        // The prepare SVD results, we'll decomp A to UxSxV'
        INDArray s = Nd4j.create(m < n ? m : n);
        INDArray VT = Nd4j.create(n, n, 'f');

        // Note - we don't care about U 
        Nd4j.getBlasWrapper().lapack().sgesvd(A, s, null, VT);

        // for comparison k & nDims are the equivalent values in both methods implementing PCA

        // So now let's rip out the appropriate number of left singular vectors from
        // the V output (note we pulls rows since VT is a transpose of V)
        INDArray V = VT.transpose();
        INDArray factor = Nd4j.create(n, nDims, 'f');
        for (int i = 0; i < nDims; i++) {
            factor.putColumn(i, V.getColumn(i));
        }

        return factor;
    }



    /**
     * Calculates pca reduced value of a matrix, for a given variance. A larger variance (99%)
     * will result in a higher order feature set.
     *
     * The returned matrix is a projection of A onto principal components
     *
     * @see pca(INDArray, int, boolean)
     *
     * @param A the array of features, rows are results, columns are features - will be changed
     * @param variance the amount of variance to preserve as a float 0 - 1
     * @param normalize whether to normalize (set features to have zero mean)
     * @return the matrix representing  a reduced feature set
     */
    public static INDArray pca(INDArray A, double variance, boolean normalize) {
        INDArray factor = pca_factor(A, variance, normalize);
        return A.mmul(factor);
    }


    /**
     * Calculates pca vectors of a matrix, for a given variance. A larger variance (99%)
     * will result in a higher order feature set.
     *
     * To use the returned factor: multiply feature(s) by the factor to get a reduced dimension
     *
     * INDArray Areduced = A.mmul( factor ) ;
     * 
     * The array Areduced is a projection of A onto principal components
     *
     * @see pca(INDArray, double, boolean)
     *
     * @param A the array of features, rows are results, columns are features - will be changed
     * @param variance the amount of variance to preserve as a float 0 - 1
     * @param normalize whether to normalize (set features to have zero mean)
     * @return the matrix to mulitiply a feature by to get a reduced feature set
     */
    public static INDArray pca_factor(INDArray A, double variance, boolean normalize) {
        if (normalize) {
            // Normalize to mean 0 for each feature ( each column has 0 mean )
            INDArray mean = A.mean(0);
            A.subiRowVector(mean);
        }

        int m = A.rows();
        int n = A.columns();

        // The prepare SVD results, we'll decomp A to UxSxV'
        INDArray s = Nd4j.create(m < n ? m : n);
        INDArray VT = Nd4j.create(n, n, 'f');

        // Note - we don't care about U 
        Nd4j.getBlasWrapper().lapack().sgesvd(A, s, null, VT);

        // Now convert the eigs of X into the eigs of the covariance matrix
        for (int i = 0; i < s.length(); i++) {
            s.putScalar(i, Math.sqrt(s.getDouble(i)) / (m - 1));
        }

        // Now find how many features we need to preserve the required variance
        // Which is the same percentage as a cumulative sum of the eigenvalues' percentages
        double totalEigSum = s.sumNumber().doubleValue() * variance;
        int k = -1; // we will reduce to k dimensions
        double runningTotal = 0;
        for (int i = 0; i < s.length(); i++) {
            runningTotal += s.getDouble(i);
            if (runningTotal >= totalEigSum) { // OK I know it's a float, but what else can we do ?
                k = i + 1; // we will keep this many features to preserve the reqd. variance
                break;
            }
        }
        if (k == -1) { // if we need everything
            throw new RuntimeException("No reduction possible for reqd. variance - use smaller variance");
        }
        // So now let's rip out the appropriate number of left singular vectors from
        // the V output (note we pulls rows since VT is a transpose of V)
        INDArray V = VT.transpose();
        INDArray factor = Nd4j.create(n, k, 'f');
        for (int i = 0; i < k; i++) {
            factor.putColumn(i, V.getColumn(i));
        }

        return factor;
    }

}
