/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * specific language governing permissions and limitations under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.graph;

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Pairwise mutual information (MI) matrix between feature columns via histogram binning.
 *
 * <p>Given a data matrix {@code X} of shape [n, d], computes the [d, d] symmetric matrix
 * where {@code MI[i, j]} is the empirical mutual information between column i and column j,
 * estimated by uniform histogram binning with {@code numBins} bins per variable.
 *
 * <p>Mutual information formula:
 * <pre>
 *   MI(X_i, X_j) = sum_{a,b} p(a,b) * log( p(a,b) / (p(a) * p(b)) )
 * </pre>
 * where {@code p(a)} and {@code p(b)} are the empirical marginal distributions and
 * {@code p(a,b)} is the empirical joint distribution, estimated by binning into a
 * {@code numBins x numBins} grid.
 *
 * <p>Diagonal entries are the self-MI (= marginal entropy), which serves as an upper bound
 * for all off-diagonal MIs. Entries are non-negative by the information-theoretic property
 * of MI; independent columns will give values close to 0.
 */
public class MutualInformationGraph {

    private MutualInformationGraph() {}

    /**
     * Compute the pairwise mutual information matrix for all pairs of columns in {@code data}.
     *
     * @param data    observations matrix [n, d] (any numeric dtype; cast to DOUBLE internally)
     * @param numBins number of histogram bins per variable (e.g. 10 or sqrt(n) is typical)
     * @return symmetric MI matrix [d, d] in nats (natural-log base)
     */
    public static INDArray compute(INDArray data, int numBins) {
        Preconditions.checkArgument(data.rank() == 2,
                "data must be rank 2 [n, d]; got shape %s", data.shapeInfoToString());
        Preconditions.checkArgument(numBins >= 2,
                "numBins must be >= 2; got %s", numBins);

        INDArray X = data.castTo(DataType.DOUBLE);
        int n = (int) X.rows();
        int d = (int) X.columns();

        // Extract all columns into double[][]
        double[][] cols = new double[d][n];
        for (int j = 0; j < d; j++) {
            for (int i = 0; i < n; i++) {
                cols[j][i] = X.getDouble(i, j);
            }
        }

        // Compute per-column bin edges: [min, max] -> numBins equal-width bins
        double[] colMin = new double[d];
        double[] colMax = new double[d];
        for (int j = 0; j < d; j++) {
            double mn = Double.MAX_VALUE, mx = -Double.MAX_VALUE;
            for (int i = 0; i < n; i++) {
                if (cols[j][i] < mn) mn = cols[j][i];
                if (cols[j][i] > mx) mx = cols[j][i];
            }
            colMin[j] = mn;
            colMax[j] = mx + 1e-12; // ensure the maximum value falls inside the last bin
        }

        // Pre-bin each column: binIdx[j][i] = bin index (0..numBins-1) for observation i of column j
        int[][] binIdx = new int[d][n];
        for (int j = 0; j < d; j++) {
            double width = (colMax[j] - colMin[j]) / numBins;
            for (int i = 0; i < n; i++) {
                int b = (int) ((cols[j][i] - colMin[j]) / width);
                if (b < 0) b = 0;
                if (b >= numBins) b = numBins - 1;
                binIdx[j][i] = b;
            }
        }

        double[] miFlat = new double[d * d];

        for (int ci = 0; ci < d; ci++) {
            for (int cj = ci; cj < d; cj++) {
                // Build joint histogram [numBins x numBins]
                long[] joint = new long[numBins * numBins];
                for (int obs = 0; obs < n; obs++) {
                    int a = binIdx[ci][obs];
                    int b = binIdx[cj][obs];
                    joint[a * numBins + b]++;
                }

                // Compute marginals
                long[] margI = new long[numBins];
                long[] margJ = new long[numBins];
                for (int a = 0; a < numBins; a++) {
                    for (int b = 0; b < numBins; b++) {
                        margI[a] += joint[a * numBins + b];
                        margJ[b] += joint[a * numBins + b];
                    }
                }

                // Compute MI = sum p(a,b) log( p(a,b) / (p(a)*p(b)) )
                double mi = 0.0;
                for (int a = 0; a < numBins; a++) {
                    for (int b = 0; b < numBins; b++) {
                        long jab = joint[a * numBins + b];
                        if (jab == 0) continue;
                        long ma = margI[a];
                        long mb = margJ[b];
                        if (ma == 0 || mb == 0) continue;
                        double pab = (double) jab / n;
                        double pa  = (double) ma  / n;
                        double pb  = (double) mb  / n;
                        mi += pab * Math.log(pab / (pa * pb));
                    }
                }
                // MI is symmetric and non-negative; clamp tiny negatives to 0
                if (mi < 0.0) mi = 0.0;

                miFlat[ci * d + cj] = mi;
                miFlat[cj * d + ci] = mi;
            }
        }

        return Nd4j.create(miFlat, new long[]{d, d}, DataType.DOUBLE);
    }
}
