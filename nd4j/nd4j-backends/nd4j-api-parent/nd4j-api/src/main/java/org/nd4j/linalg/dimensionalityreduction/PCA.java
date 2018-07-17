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

package org.nd4j.linalg.dimensionalityreduction;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * PCA class for dimensionality reduction and general analysis
 *
 * @author Adam Gibson
 * @author Luke Czapla - added methods used in non-static usage of PCA
 */
public class PCA {

    private INDArray covarianceMatrix, mean, eigenvectors, eigenvalues;

    private PCA() {}

    /**
     * Create a PCA instance with calculated data: covariance, mean, eigenvectors, and eigenvalues.
     * @param dataset The set of data (records) of features, each row is a data record and each
     *                column is a feature, every data record has the same number of features.
     */
    public PCA(INDArray dataset) {
        INDArray[] covmean = covarianceMatrix(dataset);
        this.covarianceMatrix = covmean[0];
        this.mean = covmean[1];
        INDArray[] pce = principalComponents(covmean[0]);
        this.eigenvectors = pce[0];
        this.eigenvalues = pce[1];
    }


    /**
     * Return a reduced basis set that covers a certain fraction of the variance of the data
     * @param variance The desired fractional variance (0 to 1), it will always be greater than the value.
     * @return The basis vectors as columns, size <i>N</i> rows by <i>ndims</i> columns, where <i>ndims</i> is less than or equal to <i>N</i>
     */
    public INDArray reducedBasis(double variance) {
        INDArray vars = Transforms.pow(eigenvalues, -0.5, true);
        double res = vars.sumNumber().doubleValue();
        double total = 0.0;
        int ndims = 0;
        for (int i = 0; i < vars.columns(); i++) {
            ndims++;
            total += vars.getDouble(i);
            if (total / res > variance)
                break;
        }
        INDArray result = Nd4j.create(eigenvectors.rows(), ndims);
        for (int i = 0; i < ndims; i++)
            result.putColumn(i, eigenvectors.getColumn(i));
        return result;
    }


    /**
     * Takes a set of data on each row, with the same number of features as the constructing data
     * and returns the data in the coordinates of the basis set about the mean.
     * @param data Data of the same features used to construct the PCA object
     * @return The record in terms of the principal component vectors, you can set unused ones to zero.
     */
    public INDArray convertToComponents(INDArray data) {
        INDArray dx = data.subRowVector(mean);
        return Nd4j.tensorMmul(eigenvectors.transpose(), dx, new int[][] {{1}, {1}}).transposei();
    }


    /**
     * Take the data that has been transformed to the principal components about the mean and
     * transform it back into the original feature set.  Make sure to fill in zeroes in columns
     * where components were dropped!
     * @param data Data of the same features used to construct the PCA object but as the components
     * @return The records in terms of the original features
     */
    public INDArray convertBackToFeatures(INDArray data) {
        return Nd4j.tensorMmul(eigenvectors, data, new int[][] {{1}, {1}}).transposei().addiRowVector(mean);
    }


    /**
     * Estimate the variance of a single record with reduced # of dimensions.
     * @param data A single record with the same <i>N</i> features as the constructing data set
     * @param ndims The number of dimensions to include in calculation
     * @return The fraction (0 to 1) of the total variance covered by the <i>ndims</i> basis set.
     */
    public double estimateVariance(INDArray data, int ndims) {
        INDArray dx = data.sub(mean);
        INDArray v = eigenvectors.transpose().mmul(dx.reshape(dx.columns(), 1));
        INDArray t2 = Transforms.pow(v, 2);
        double fraction = t2.get(NDArrayIndex.interval(0, ndims)).sumNumber().doubleValue();
        double total = t2.sumNumber().doubleValue();
        return fraction / total;
    }


    /**
     * Generates a set of <i>count</i> random samples with the same variance and mean and eigenvector/values
     * as the data set used to initialize the PCA object, with same number of features <i>N</i>.
     * @param count The number of samples to generate
     * @return A matrix of size <i>count</i> rows by <i>N</i> columns
     */
    public INDArray generateGaussianSamples(long count) {
        INDArray samples = Nd4j.randn(new long[] {count, eigenvalues.columns()});
        INDArray factors = Transforms.pow(eigenvalues, -0.5, true);
        samples.muliRowVector(factors);
        return Nd4j.tensorMmul(eigenvectors, samples, new int[][] {{1}, {1}}).transposei().addiRowVector(mean);
    }


    /**
     * Calculates pca vectors of a matrix, for a flags number of reduced features
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
     * Calculates pca factors of a matrix, for a flags number of reduced features
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

        long m = A.rows();
        long n = A.columns();

        // The prepare SVD results, we'll decomp A to UxSxV'
        INDArray s = Nd4j.create(m < n ? m : n);
        INDArray VT = Nd4j.create(n, n, 'f');

        // Note - we don't care about U 
        Nd4j.getBlasWrapper().lapack().gesvd(A, s, null, VT);

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

        long m = A.rows();
        long n = A.columns();

        // The prepare SVD results, we'll decomp A to UxSxV'
        INDArray s = Nd4j.create(m < n ? m : n);
        INDArray VT = Nd4j.create(n, n, 'f');

        // Note - we don't care about U 
        Nd4j.getBlasWrapper().lapack().gesvd(A, s, null, VT);

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


    /**
     * This method performs a dimensionality reduction, including principal components
     * that cover a fraction of the total variance of the system.  It does all calculations
     * about the mean.
     * @param in A matrix of datapoints as rows, where column are features with fixed number N
     * @param variance The desired fraction of the total variance required
     * @return The reduced basis set
     */
    public static INDArray pca2(INDArray in, double variance) {
        // let's calculate the covariance and the mean
        INDArray[] covmean = covarianceMatrix(in);
        // use the covariance matrix (inverse) to find "force constants" and then break into orthonormal
        // unit vector components
        INDArray[] pce = principalComponents(covmean[0]);
        // calculate the variance of each component
        INDArray vars = Transforms.pow(pce[1], -0.5, true);
        double res = vars.sumNumber().doubleValue();
        double total = 0.0;
        int ndims = 0;
        for (int i = 0; i < vars.columns(); i++) {
            ndims++;
            total += vars.getDouble(i);
            if (total / res > variance)
                break;
        }
        INDArray result = Nd4j.create(in.columns(), ndims);
        for (int i = 0; i < ndims; i++)
            result.putColumn(i, pce[0].getColumn(i));
        return result;
    }

    /**
     * Returns the covariance matrix of a data set of many records, each with N features.
     * It also returns the average values, which are usually going to be important since in this
     * version, all modes are centered around the mean.  It's a matrix that has elements that are
     * expressed as average dx_i * dx_j (used in procedure) or average x_i * x_j - average x_i * average x_j
     *
     * @param in A matrix of vectors of fixed length N (N features) on each row
     * @return INDArray[2], an N x N covariance matrix is element 0, and the average values is element 1.
     */
    public static INDArray[] covarianceMatrix(INDArray in) {
        long dlength = in.rows();
        long vlength = in.columns();

        INDArray sum = Nd4j.create(vlength);
        INDArray product = Nd4j.create(vlength, vlength);

        for (int i = 0; i < vlength; i++)
            sum.getColumn(i).assign(in.getColumn(i).sumNumber().doubleValue() / dlength);

        for (int i = 0; i < dlength; i++) {
            INDArray dx1 = in.getRow(i).sub(sum);
            product.addi(dx1.reshape(vlength, 1).mmul(dx1.reshape(1, vlength)));
        }
        product.divi(dlength);
        return new INDArray[] {product, sum};
    }

    /**
     * Calculates the principal component vectors and their eigenvalues (lambda) for the covariance matrix.
     * The result includes two things: the eigenvectors (modes) as result[0] and the eigenvalues (lambda)
     * as result[1].
     *
     * @param cov The covariance matrix (calculated with the covarianceMatrix(in) method)
     * @return Array INDArray[2] "result".  The principal component vectors in decreasing flexibility are
     *      the columns of element 0 and the eigenvalues are element 1.
     */
    public static INDArray[] principalComponents(INDArray cov) {
        assert cov.rows() == cov.columns();
        INDArray[] result = new INDArray[2];
        result[0] = Nd4j.eye(cov.rows());
        result[1] = Eigen.symmetricGeneralizedEigenvalues(result[0], cov, true);
        return result;
    }


    public INDArray getCovarianceMatrix() {
        return covarianceMatrix;
    }

    public INDArray getMean() {
        return mean;
    }

    public INDArray getEigenvectors() {
        return eigenvectors;
    }

    public INDArray getEigenvalues() {
        return eigenvalues;
    }

}
