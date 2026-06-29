/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.graph;

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SparseFormat;
import org.nd4j.linalg.api.ndarray.SparseNDArray;
import org.nd4j.linalg.api.ops.impl.sparse.Semiring;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Classical graph algorithms implemented via GraphBLAS-style semiring SpMV/SpMM operations
 * ({@link SparseNDArray#mvSemiring} and {@link SparseNDArray#mmulSemiring}).
 *
 * <p>All methods require the adjacency matrix to be in {@link SparseFormat#CSR} format.
 * Convert with {@link SparseNDArray#toCsr()} if needed.
 *
 * <h3>Algorithm — semiring mapping</h3>
 * <table>
 *   <tr><th>Method</th><th>Semiring</th><th>Description</th></tr>
 *   <tr><td>{@link #bfs}</td><td>OR_AND</td>
 *       <td>Frontier propagation: vertex is reached if any in-neighbour was in the frontier</td></tr>
 *   <tr><td>{@link #sssp}</td><td>MIN_PLUS</td>
 *       <td>Relax: new dist to i = min over neighbours k of (weight(k,i) + dist(k))</td></tr>
 *   <tr><td>{@link #connectedComponents}</td><td>MAX_PLUS with zero-weight adj</td>
 *       <td>Label propagation: each vertex takes the max label of its neighbourhood</td></tr>
 *   <tr><td>{@link #pageRank}</td><td>PLUS_TIMES</td>
 *       <td>Standard power iteration on the column-stochastic transition matrix</td></tr>
 * </table>
 */
public final class GraphAlgorithms {

    private GraphAlgorithms() {}

    // -----------------------------------------------------------------------
    // BFS
    // -----------------------------------------------------------------------

    /**
     * Breadth-first search via repeated OR_AND SpMV until fixpoint.
     *
     * <p>Each iteration propagates the current frontier one hop through the adjacency
     * matrix using the OR_AND semiring:
     * <pre>
     *   newFrontier[i] = OR_{k : (i,k) in adj} (adj[i,k] AND frontier[k])
     * </pre>
     * Since adjacency values are 1.0 (edges), AND(1.0, frontier[k]) = frontier[k], so
     * a vertex enters the new frontier if any of its in-neighbours was in the previous
     * frontier.  Vertices already visited are masked out before updating {@code visited}.
     *
     * <p>Complexity: O(diameter * nnz) SpMV iterations.
     *
     * @param adj    adjacency matrix in CSR format; values should be 1.0 for edges
     * @param source 0-based source vertex index
     * @return dense 1D array of length n where {@code result[i] = 1.0} if vertex i is
     *         reachable from {@code source}, {@code 0.0} otherwise
     * @throws IllegalStateException    if {@code adj} is not in CSR format
     * @throws IllegalArgumentException if {@code source} is out of range
     */
    public static INDArray bfs(SparseNDArray adj, int source) {
        Preconditions.checkState(adj.getFormat() == SparseFormat.CSR,
                "bfs() requires CSR format; call adj.toCsr() first. Current format: " + adj.getFormat());
        int n = (int) adj.rows();
        Preconditions.checkArgument(source >= 0 && source < n,
                "source vertex %d is out of range [0, %d)", source, n);

        // Frontier: one-hot at source vertex
        INDArray frontier = Nd4j.zeros(adj.dataType(), n);
        frontier.putScalar(source, 1.0);
        INDArray visited = frontier.dup();

        // Iterate: one BFS level per iteration, at most n levels in the worst case
        for (int iter = 0; iter < n; iter++) {
            // newFrontier[i] = 1 if any in-neighbour of i was in the frontier
            INDArray newFrontier = adj.mvSemiring(frontier, Semiring.OR_AND);

            // Keep only vertices that are newly reached (not yet in visited)
            // added[i] = newFrontier[i] * (1 - visited[i])
            INDArray added = newFrontier.mul(visited.rsub(1.0));

            // Convergence: no new vertices reached
            if (added.maxNumber().doubleValue() == 0.0) {
                break;
            }

            visited = visited.add(added);
            frontier = added;
        }
        return visited;
    }

    // -----------------------------------------------------------------------
    // Single-source shortest paths (Bellman-Ford / MIN_PLUS)
    // -----------------------------------------------------------------------

    /**
     * Single-source shortest paths via Bellman-Ford iteration using the MIN_PLUS semiring.
     *
     * <p>Each iteration applies one round of edge relaxation:
     * <pre>
     *   newDist[i] = min_{k : (i,k) in adj} (weight(i,k) + dist[k])
     *   dist[i]    = min(dist[i], newDist[i])
     * </pre>
     * Convergence is detected when no distance changes, or after at most {@code n-1} rounds
     * (Bellman-Ford guarantee for graphs with no negative cycles).
     *
     * <p>Unreachable vertices are returned as {@link Float#POSITIVE_INFINITY} cast to the
     * matrix dtype.
     *
     * @param adj    weighted adjacency matrix in CSR format; values are edge weights
     * @param source 0-based source vertex index
     * @return dense 1D array of length n where {@code result[i]} is the shortest-path
     *         distance from {@code source} to vertex i ({@code +Inf} if unreachable)
     * @throws IllegalStateException    if {@code adj} is not in CSR format
     * @throws IllegalArgumentException if {@code source} is out of range
     */
    public static INDArray sssp(SparseNDArray adj, int source) {
        Preconditions.checkState(adj.getFormat() == SparseFormat.CSR,
                "sssp() requires CSR format; call adj.toCsr() first. Current format: " + adj.getFormat());
        int n = (int) adj.rows();
        Preconditions.checkArgument(source >= 0 && source < n,
                "source vertex %d is out of range [0, %d)", source, n);

        // Initialise: distance = +Inf everywhere except source = 0
        float[] distData = new float[n];
        java.util.Arrays.fill(distData, Float.POSITIVE_INFINITY);
        distData[source] = 0.0f;
        INDArray dist = Nd4j.createFromArray(distData).castTo(adj.dataType());

        // Bellman-Ford: at most n-1 relaxation rounds (exit early if no change)
        for (int iter = 0; iter < n - 1; iter++) {
            // newDist[i] = min_k ( weight(i,k) + dist[k] )  for k in row i's neighbours
            INDArray newDist = adj.mvSemiring(dist, Semiring.MIN_PLUS);

            // updated[i] = min(newDist[i], dist[i])
            INDArray updated = Nd4j.where(newDist.lt(dist), newDist, dist)[0];

            // Convergence check: no distance improved
            if (updated.equalsWithEps(dist, 1e-9f)) {
                break;
            }
            dist = updated;
        }
        return dist;
    }

    // -----------------------------------------------------------------------
    // Connected components (label propagation / MAX_PLUS with zero-weight adj)
    // -----------------------------------------------------------------------

    /**
     * Connected components via label propagation using the MAX_PLUS semiring.
     *
     * <p>Each vertex starts with its own index as its component label.  In each iteration,
     * every vertex adopts the maximum label seen in its neighbourhood:
     * <pre>
     *   newLabel[i] = max_{k : (i,k) in adj} labels[k]   (MAX_PLUS with edge weight 0)
     *   label[i]    = max(label[i], newLabel[i])          (keep own label if larger)
     * </pre>
     * Because the largest vertex index in each component propagates outward, all vertices
     * in the same connected component converge to the same (maximum) label within at most
     * {@code diameter} iterations.  Convergence across all components is guaranteed within
     * n iterations.
     *
     * <p><b>Implementation note:</b> the input adjacency is scaled to all-zero values
     * (preserving sparsity structure) before applying MAX_PLUS.  With edge weight 0:
     * {@code mul(0, label[k]) = 0 + label[k] = label[k]}, so the semiring correctly
     * propagates labels rather than sums them.
     *
     * @param adj undirected adjacency matrix in CSR format (should be symmetric; values
     *            can be any non-zero — they are replaced internally with zeros)
     * @return dense 1D array of length n where {@code result[i]} is the component label
     *         (the maximum vertex index in the component containing i)
     * @throws IllegalStateException if {@code adj} is not in CSR format
     */
    public static INDArray connectedComponents(SparseNDArray adj) {
        Preconditions.checkState(adj.getFormat() == SparseFormat.CSR,
                "connectedComponents() requires CSR format; call adj.toCsr() first. Current format: "
                        + adj.getFormat());
        int n = (int) adj.rows();

        // Initial label for each vertex = its own index
        double[] labelData = new double[n];
        for (int i = 0; i < n; i++) {
            labelData[i] = i;
        }
        INDArray labels = Nd4j.createFromArray(labelData).castTo(adj.dataType());

        // Zero-weight adjacency: same sparsity as adj, but all values = 0.
        // With MAX_PLUS, mul(0, label[j]) = 0 + label[j] = label[j], so the semiring
        // computes max over neighbour labels — exactly what label propagation needs.
        SparseNDArray zeroAdj = adj.scale(0.0);

        for (int iter = 0; iter < n; iter++) {
            // newLabels[i] = max label among neighbours of i; -Inf for isolated vertices
            INDArray newLabels = zeroAdj.mvSemiring(labels, Semiring.MAX_PLUS);

            // Merge: take max of own label and best-neighbour label
            // This ensures isolated vertices keep their own label (max(label, -Inf) = label)
            INDArray merged = Nd4j.where(labels.gt(newLabels), labels, newLabels)[0];

            // Convergence: no label changed
            if (merged.equalsWithEps(labels, 1e-9f)) {
                break;
            }
            labels = merged;
        }
        return labels;
    }

    // -----------------------------------------------------------------------
    // PageRank (power iteration / PLUS_TIMES)
    // -----------------------------------------------------------------------

    /**
     * PageRank via power iteration using the PLUS_TIMES semiring.
     *
     * <p>The transition matrix is derived from {@code adj} by column-normalising each
     * column by its out-degree (column sum), so that following any edge from a vertex is
     * equally likely.  The standard random-walk formula is then iterated:
     * <pre>
     *   rank_{t+1} = damping * (AN · rank_t) + (1 - damping) / n * ones
     * </pre>
     * where {@code AN} is the column-stochastic form of the adjacency.  Dangling nodes
     * (columns whose sum is zero) are left as zero columns so their "lost" rank dissipates.
     *
     * <p>The result is always non-negative and sums to approximately 1.0 after convergence.
     *
     * @param adj     adjacency matrix in CSR format ({@code A[i,j] != 0} means there is an
     *                edge from j to i; typical convention for column-stochastic transition)
     * @param damping damping factor, typically 0.85; must be in (0, 1)
     * @param iters   number of power iterations to perform
     * @return dense 1D array of length n containing PageRank scores (sum &asymp; 1)
     * @throws IllegalStateException    if {@code adj} is not in CSR format
     * @throws IllegalArgumentException if {@code damping} is outside (0, 1) or {@code iters &le; 0}
     */
    public static INDArray pageRank(SparseNDArray adj, double damping, int iters) {
        Preconditions.checkState(adj.getFormat() == SparseFormat.CSR,
                "pageRank() requires CSR format; call adj.toCsr() first. Current format: " + adj.getFormat());
        Preconditions.checkArgument(damping > 0.0 && damping < 1.0,
                "damping must be in (0, 1), got %s", damping);
        Preconditions.checkArgument(iters > 0,
                "iters must be positive, got %d", iters);

        int n = (int) adj.rows();

        // Materialise adj to dense to compute column sums and normalise
        INDArray A = adj.toDense();

        // Column sums: A.sum(0) gives a [1, cols] array; squeeze to [cols]
        INDArray colSums = A.sum(0);

        // Zero-safe reciprocal: dangling nodes (colSum == 0) get weight 0, not Inf
        INDArray safeColSums = Nd4j.where(colSums.neq(0),
                colSums,
                Nd4j.ones(adj.dataType(), colSums.length()))[0];
        INDArray AN = A.divRowVector(safeColSums);   // column-normalise: AN[i,j] / colSum[j]

        // Convert back to CSR sparse for efficient SpMV in the iteration loop
        SparseNDArray spAN = Nd4j.toSparse(AN, SparseFormat.CSR);

        // Uniform initial rank vector
        INDArray rank = Nd4j.ones(adj.dataType(), n).divi(n);

        // Teleportation vector: (1 - damping) / n  (broadcast constant)
        double tele = (1.0 - damping) / n;
        INDArray teleVec = Nd4j.valueArrayOf(new long[]{n}, tele).castTo(adj.dataType());

        for (int iter = 0; iter < iters; iter++) {
            // rank = damping * (AN * rank) + teleVec
            rank = spAN.mvSemiring(rank, Semiring.PLUS_TIMES).muli(damping).addi(teleVec);
        }
        return rank;
    }
}
