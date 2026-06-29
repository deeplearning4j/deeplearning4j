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

package org.eclipse.deeplearning4j.nd4j.linalg.sparse;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SparseFormat;
import org.nd4j.linalg.api.ndarray.SparseNDArray;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSpmvSemiring;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSpmmSemiring;
import org.nd4j.linalg.api.ops.impl.sparse.Semiring;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for GraphBLAS-style semiring sparse matrix operations and graph
 * algorithms built on top of the CSR SpMV / SpMM semiring ops.
 *
 * <p>Graph algorithm helpers (BFS, SSSP, connected components, PageRank) are
 * implemented inline in this class using {@link CsrSpmvSemiring}.  They are
 * intended as the specification for a future {@code org.nd4j.linalg.graph.GraphAlgorithms}
 * facade class; once that class exists the helpers below can be replaced by
 * calls to it.
 *
 * <p>All tests use {@link DataType#FLOAT} and are expected to pass on both
 * CPU and CUDA backends.
 */
@DisplayName("GraphBLAS Semiring SpMV/SpMM and Graph Algorithm Tests")
public class SparseGraphAlgoTest {

    private static final double TOL = 1e-4;

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    /**
     * Assert element-wise equality within {@code tol}.  Handles
     * {@code Float.POSITIVE_INFINITY} exactly (Inf == Inf).
     */
    private static void assertClose(INDArray expected, INDArray actual, double tol, String msg) {
        assertEquals(expected.length(), actual.length(), msg + ": length mismatch");
        for (long i = 0; i < expected.length(); i++) {
            double e = expected.getDouble(i);
            double a = actual.getDouble(i);
            if (Double.isInfinite(e)) {
                assertTrue(Double.isInfinite(a) && Math.signum(e) == Math.signum(a),
                        msg + ": expected Inf at index " + i + " but got " + a);
            } else {
                assertEquals(e, a, tol, msg + ": mismatch at index " + i);
            }
        }
    }

    /**
     * Convert a row-major 2-D double array to a CSR {@link SparseNDArray}
     * (FLOAT dtype) via {@link Nd4j#toSparse}.
     */
    private static SparseNDArray denseToSparse(double[][] data) {
        int rows = data.length, cols = data[0].length;
        INDArray dense = Nd4j.zeros(DataType.FLOAT, rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                dense.putScalar(new int[]{r, c}, (float) data[r][c]);
        return Nd4j.toSparse(dense, SparseFormat.CSR);
    }

    // -----------------------------------------------------------------------
    // Test 1 — PLUS_TIMES SpMV (standard matrix-vector multiply)
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("Semiring SpMV PLUS_TIMES — hand-computed 3x3 * [3] verification")
    public void testSpmvPlusTimes() {
        // A = [[2, 0, 1],   x = [1, 2, 3]
        //      [0, 3, 0],
        //      [4, 0, 2]]
        // y = [2*1 + 1*3, 3*2, 4*1 + 2*3] = [5, 6, 10]
        SparseNDArray A = denseToSparse(new double[][]{
            {2, 0, 1},
            {0, 3, 0},
            {4, 0, 2}
        });
        INDArray x = Nd4j.createFromArray(new float[]{1f, 2f, 3f});
        INDArray expected = Nd4j.createFromArray(new float[]{5f, 6f, 10f});

        CsrSpmvSemiring op = new CsrSpmvSemiring(
                A.getValues(), A.getColIdx(), A.getRowPtr(), x,
                3L, 3L, Semiring.PLUS_TIMES);
        INDArray actual = Nd4j.exec(op)[0];

        assertClose(expected, actual, TOL, "PLUS_TIMES SpMV 3x3");
    }

    // -----------------------------------------------------------------------
    // Test 2 — MIN_PLUS SpMV (one Bellman-Ford relaxation step from source 0)
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("Semiring SpMV MIN_PLUS — single Bellman-Ford step from source vertex 0")
    public void testSpmvMinPlus() {
        // Undirected weight matrix (only finite entries stored as CSR):
        //   A[0,0]=1  A[0,2]=2     (row 0 nnz: cols 0,2)
        //   A[1,2]=1               (row 1 nnz: col 2)
        //   A[2,0]=2  A[2,1]=1    (row 2 nnz: cols 0,1)
        //
        // x = [0, INF, INF]   (distance vector, source = vertex 0)
        //
        // out[i] = min_j( A[i,j] + x[j] )
        //   out[0] = min(1+0, 2+INF)       = 1
        //   out[1] = min(1+INF)            = INF
        //   out[2] = min(2+0, 1+INF)       = 2
        INDArray values = Nd4j.createFromArray(new float[]{1f, 2f, 1f, 2f, 1f});
        INDArray colIdx = Nd4j.createFromArray(new int[]{0, 2,   2,   0, 1});
        INDArray rowPtr = Nd4j.createFromArray(new int[]{0, 2, 3, 5});
        SparseNDArray A = new SparseNDArray(values, colIdx, rowPtr,
                new long[]{3, 3}, SparseFormat.CSR);

        INDArray x = Nd4j.createFromArray(new float[]{0f, Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY});

        CsrSpmvSemiring op = new CsrSpmvSemiring(
                A.getValues(), A.getColIdx(), A.getRowPtr(), x,
                3L, 3L, Semiring.MIN_PLUS);
        INDArray actual = Nd4j.exec(op)[0];

        assertEquals(1f, actual.getFloat(0), (float) TOL, "MIN_PLUS out[0]");
        assertTrue(Float.isInfinite(actual.getFloat(1)), "MIN_PLUS out[1] should be INF");
        assertEquals(2f, actual.getFloat(2), (float) TOL, "MIN_PLUS out[2]");
    }

    // -----------------------------------------------------------------------
    // Test 3 — MAX_PLUS SpMV
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("Semiring SpMV MAX_PLUS — 2x3 * [3] hand-computed")
    public void testSpmvMaxPlus() {
        // Sparse 2x3: A[0,1]=1.0,  A[1,0]=2.0,  A[1,2]=0.5
        // x = [3.0, 1.0, 2.0]
        // out[0] = max(1.0+1.0)              = 2.0
        // out[1] = max(2.0+3.0, 0.5+2.0)    = max(5.0, 2.5) = 5.0
        INDArray values = Nd4j.createFromArray(new float[]{1.0f, 2.0f, 0.5f});
        INDArray colIdx = Nd4j.createFromArray(new int[]{1,   0, 2});
        INDArray rowPtr = Nd4j.createFromArray(new int[]{0, 1, 3});
        SparseNDArray A = new SparseNDArray(values, colIdx, rowPtr,
                new long[]{2, 3}, SparseFormat.CSR);

        INDArray x = Nd4j.createFromArray(new float[]{3.0f, 1.0f, 2.0f});

        CsrSpmvSemiring op = new CsrSpmvSemiring(
                A.getValues(), A.getColIdx(), A.getRowPtr(), x,
                2L, 3L, Semiring.MAX_PLUS);
        INDArray actual = Nd4j.exec(op)[0];

        assertEquals(2.0f, actual.getFloat(0), (float) TOL, "MAX_PLUS out[0]");
        assertEquals(5.0f, actual.getFloat(1), (float) TOL, "MAX_PLUS out[1]");
    }

    // -----------------------------------------------------------------------
    // BFS inline helper — OR_AND semiring frontier propagation
    // -----------------------------------------------------------------------

    /**
     * BFS reachability from {@code source} using the OR_AND semiring.
     *
     * <p>Each iteration propagates the frontier one hop:
     * {@code newFrontier[i] = OR_j( A[i,j] AND frontier[j] )}
     * which in float is {@code max_j( min(A[i,j], frontier[j]) )}.
     * A 1.0 entry in the returned array means the vertex is reachable.
     *
     * <p>This is the specification for {@code GraphAlgorithms.bfs(adj, source)}.
     */
    private static float[] bfs(SparseNDArray adj, int n, int source) {
        float[] visited  = new float[n];
        float[] frontier = new float[n];
        visited[source]  = 1f;
        frontier[source] = 1f;

        for (int iter = 0; iter < n; iter++) {
            INDArray fVec = Nd4j.createFromArray(frontier);
            CsrSpmvSemiring op = new CsrSpmvSemiring(
                    adj.getValues(), adj.getColIdx(), adj.getRowPtr(), fVec,
                    n, n, Semiring.OR_AND);
            INDArray nextArr = Nd4j.exec(op)[0];

            boolean anyNew = false;
            for (int i = 0; i < n; i++) {
                boolean candidate = nextArr.getFloat(i) > 0f;
                boolean fresh     = visited[i] == 0f;
                frontier[i] = (candidate && fresh) ? 1f : 0f;
                if (candidate && fresh) {
                    visited[i] = 1f;
                    anyNew = true;
                }
            }
            if (!anyNew) break;
        }
        return visited;
    }

    // -----------------------------------------------------------------------
    // Test 4 — BFS reachability via OR_AND semiring
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("BFS OR_AND — linear chain 0-1-2-3 is fully reachable from vertex 0")
    public void testBfsLinearChainFullyReachable() {
        // Undirected: 0-1, 1-2, 2-3  (4 vertices)
        // Symmetric CSR adjacency (each undirected edge stored twice):
        //   row 0: col 1            rowPtr[0..1] = [0,1]
        //   row 1: col 0, col 2     rowPtr[1..2] = [1,3]
        //   row 2: col 1, col 3     rowPtr[2..3] = [3,5]
        //   row 3: col 2            rowPtr[3..4] = [5,6]
        INDArray values = Nd4j.createFromArray(new float[]{1f, 1f, 1f, 1f, 1f, 1f});
        INDArray colIdx = Nd4j.createFromArray(new int[]{1,   0, 2,   1, 3,   2});
        INDArray rowPtr = Nd4j.createFromArray(new int[]{0, 1, 3, 5, 6});
        SparseNDArray adj = new SparseNDArray(values, colIdx, rowPtr,
                new long[]{4, 4}, SparseFormat.CSR);

        float[] reached = bfs(adj, 4, 0);

        for (int i = 0; i < 4; i++) {
            assertEquals(1f, reached[i], (float) TOL, "BFS chain: vertex " + i + " should be reachable");
        }
    }

    @Test
    @DisplayName("BFS OR_AND — disconnected graph 0-1 | 2-3: BFS from 0 does not reach 2 or 3")
    public void testBfsDisconnectedGraph() {
        // Two separate edges: 0-1 and 2-3 (4 vertices, no cross-component edges)
        //   row 0: col 1    row 1: col 0    row 2: col 3    row 3: col 2
        INDArray values = Nd4j.createFromArray(new float[]{1f, 1f, 1f, 1f});
        INDArray colIdx = Nd4j.createFromArray(new int[]{1,   0,   3,   2});
        INDArray rowPtr = Nd4j.createFromArray(new int[]{0, 1, 2, 3, 4});
        SparseNDArray adj = new SparseNDArray(values, colIdx, rowPtr,
                new long[]{4, 4}, SparseFormat.CSR);

        float[] reached = bfs(adj, 4, 0);

        assertEquals(1f, reached[0], (float) TOL, "vertex 0 reachable (source)");
        assertEquals(1f, reached[1], (float) TOL, "vertex 1 reachable via 0-1");
        assertEquals(0f, reached[2], (float) TOL, "vertex 2 NOT reachable from component {0,1}");
        assertEquals(0f, reached[3], (float) TOL, "vertex 3 NOT reachable from component {0,1}");
    }

    // -----------------------------------------------------------------------
    // SSSP inline helper — MIN_PLUS Bellman-Ford
    // -----------------------------------------------------------------------

    /**
     * Single-source shortest paths via Bellman-Ford using the MIN_PLUS semiring.
     *
     * <p>The input adjacency {@code adj} must be stored in <em>incoming-edge</em>
     * orientation: {@code adj[i,j]} = weight of the directed edge {@code j → i}.
     * Each iteration applies:
     * {@code d_new[i] = min_j( adj[i,j] + d[j] )}
     * and then {@code d = element-wise min(d, d_new)}.  After {@code n-1}
     * iterations the result is exact for graphs without negative cycles.
     *
     * <p>This is the specification for {@code GraphAlgorithms.sssp(adj, source)}.
     */
    private static float[] sssp(SparseNDArray adj, int n, int source) {
        float[] d = new float[n];
        for (int i = 0; i < n; i++) d[i] = Float.POSITIVE_INFINITY;
        d[source] = 0f;

        for (int iter = 0; iter < n - 1; iter++) {
            INDArray dVec = Nd4j.createFromArray(d);
            CsrSpmvSemiring op = new CsrSpmvSemiring(
                    adj.getValues(), adj.getColIdx(), adj.getRowPtr(), dVec,
                    n, n, Semiring.MIN_PLUS);
            INDArray relaxed = Nd4j.exec(op)[0];

            boolean changed = false;
            for (int i = 0; i < n; i++) {
                float r = relaxed.getFloat(i);
                if (r < d[i]) { d[i] = r; changed = true; }
            }
            if (!changed) break;
        }
        return d;
    }

    // -----------------------------------------------------------------------
    // Test 5 — SSSP known shortest paths
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("SSSP MIN_PLUS Bellman-Ford — directed 4-vertex graph, exact distances from 0")
    public void testSsspKnownDistances() {
        // Directed edges:  0→1 w=4,  0→2 w=1,  2→1 w=2,  1→3 w=1,  2→3 w=5
        // Shortest paths from 0:
        //   dist[0]=0, dist[1]=3 (0→2→1), dist[2]=1 (0→2), dist[3]=4 (0→2→1→3)
        //
        // CSR stored in incoming-edge orientation: adj[i,j] = weight of j→i
        //   row 0: {} (nothing arrives at 0 in this graph)
        //   row 1: {j=0: w=4, j=2: w=2}   (0→1 and 2→1)
        //   row 2: {j=0: w=1}              (0→2)
        //   row 3: {j=1: w=1, j=2: w=5}   (1→3 and 2→3)
        INDArray values = Nd4j.createFromArray(new float[]{4f, 2f,   1f,   1f, 5f});
        INDArray colIdx = Nd4j.createFromArray(new int[]{0, 2,     0,     1, 2});
        INDArray rowPtr = Nd4j.createFromArray(new int[]{0, 0, 2, 3, 5});
        SparseNDArray adj = new SparseNDArray(values, colIdx, rowPtr,
                new long[]{4, 4}, SparseFormat.CSR);

        float[] dist = sssp(adj, 4, 0);

        assertEquals(0f, dist[0], (float) TOL, "dist[0]");
        assertEquals(3f, dist[1], (float) TOL, "dist[1] via 0→2→1");
        assertEquals(1f, dist[2], (float) TOL, "dist[2] via 0→2");
        assertEquals(4f, dist[3], (float) TOL, "dist[3] via 0→2→1→3");
    }

    // -----------------------------------------------------------------------
    // Connected-components inline helper — repeated BFS
    // -----------------------------------------------------------------------

    /**
     * Label each vertex with the index of the first (lowest-index) vertex
     * in its connected component.  Uses {@link #bfs} internally.
     *
     * <p>This is the specification for {@code GraphAlgorithms.connectedComponents(adj)}.
     */
    private static int[] connectedComponents(SparseNDArray adj, int n) {
        int[] label    = new int[n];
        boolean[] seen = new boolean[n];
        java.util.Arrays.fill(label, -1);

        for (int seed = 0; seed < n; seed++) {
            if (seen[seed]) continue;
            float[] reached = bfs(adj, n, seed);
            for (int i = 0; i < n; i++) {
                if (reached[i] > 0f) {
                    seen[i]  = true;
                    label[i] = seed;
                }
            }
        }
        return label;
    }

    // -----------------------------------------------------------------------
    // Test 6 — Connected components, 2-component graph
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("Connected components — 5 vertices, 2 components {0,1,2} and {3,4}")
    public void testConnectedComponents() {
        // Undirected edges: 0-1, 1-2 (component A) and 3-4 (component B)
        // 5 vertices total, no cross-component edges
        //   row 0: col 1
        //   row 1: col 0, col 2
        //   row 2: col 1
        //   row 3: col 4
        //   row 4: col 3
        INDArray values = Nd4j.createFromArray(new float[]{1f, 1f, 1f, 1f, 1f, 1f});
        INDArray colIdx = Nd4j.createFromArray(new int[]{1,   0, 2,   1,   4,   3});
        INDArray rowPtr = Nd4j.createFromArray(new int[]{0, 1, 3, 4, 5, 6});
        SparseNDArray adj = new SparseNDArray(values, colIdx, rowPtr,
                new long[]{5, 5}, SparseFormat.CSR);

        int[] label = connectedComponents(adj, 5);

        // Vertices 0, 1, 2 must all carry the same label.
        assertEquals(label[0], label[1], "vertices 0 and 1 in same component");
        assertEquals(label[0], label[2], "vertices 0 and 2 in same component");

        // Vertices 3 and 4 must share a label.
        assertEquals(label[3], label[4], "vertices 3 and 4 in same component");

        // The two component labels must be distinct.
        assertTrue(label[0] != label[3],
                "components {0,1,2} and {3,4} must have different labels");
    }

    // -----------------------------------------------------------------------
    // PageRank inline helper — PLUS_TIMES SpMV power iteration
    // -----------------------------------------------------------------------

    /**
     * Compute PageRank via power iteration using the PLUS_TIMES semiring.
     *
     * <p>{@code adj} must be column-stochastic CSR: {@code adj[i,j]} = probability
     * of jumping from vertex {@code j} to vertex {@code i}.  The update rule is:
     * {@code r_new = alpha * SpMV(adj, r) + (1-alpha)/n * ones}
     *
     * <p>This is the specification for {@code GraphAlgorithms.pageRank(adj, alpha, iters)}.
     */
    private static float[] pageRank(SparseNDArray adj, int n, float alpha, int maxIter) {
        float[] r = new float[n];
        for (int i = 0; i < n; i++) r[i] = 1f / n;

        float teleport = (1f - alpha) / n;

        for (int iter = 0; iter < maxIter; iter++) {
            INDArray rVec = Nd4j.createFromArray(r);
            CsrSpmvSemiring op = new CsrSpmvSemiring(
                    adj.getValues(), adj.getColIdx(), adj.getRowPtr(), rVec,
                    n, n, Semiring.PLUS_TIMES);
            INDArray propagated = Nd4j.exec(op)[0];

            for (int i = 0; i < n; i++) {
                r[i] = alpha * propagated.getFloat(i) + teleport;
            }
        }
        return r;
    }

    // -----------------------------------------------------------------------
    // Test 7 — PageRank on a uniform cycle: all ranks ≈ 1/3
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("PageRank PLUS_TIMES — symmetric 3-cycle: all ranks ≈ 1/3, sum ≈ 1")
    public void testPageRankCycle3() {
        // Directed cycle: 0→1, 1→2, 2→0
        // Column-stochastic CSR: adj[i,j] = 1 if j→i (each vertex has out-degree 1)
        //   adj[1,0]=1  (0→1)  → row 1, col 0
        //   adj[2,1]=1  (1→2)  → row 2, col 1
        //   adj[0,2]=1  (2→0)  → row 0, col 2
        //
        // By symmetry, all PageRank scores converge to 1/3.
        INDArray values = Nd4j.createFromArray(new float[]{1f,   1f,   1f});
        INDArray colIdx = Nd4j.createFromArray(new int[]{2,    0,    1});
        INDArray rowPtr = Nd4j.createFromArray(new int[]{0, 1, 2, 3});
        SparseNDArray adj = new SparseNDArray(values, colIdx, rowPtr,
                new long[]{3, 3}, SparseFormat.CSR);

        float[] rank = pageRank(adj, 3, 0.85f, 100);

        double sum = 0;
        for (float v : rank) sum += v;
        assertEquals(1.0, sum, 1e-3, "PageRank scores must sum to 1");

        for (int i = 0; i < 3; i++) {
            assertEquals(1.0 / 3.0, rank[i], 0.01,
                    "PageRank rank[" + i + "] should be ≈ 1/3 for uniform 3-cycle");
        }
    }

    // -----------------------------------------------------------------------
    // Test 8 — Semiring SpMM MIN_PLUS (small matrix * dense matrix)
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("Semiring SpMM MIN_PLUS — 3x3 sparse A * 3x3 dense B, hand-computed")
    public void testSpmmMinPlus() {
        // Sparse A (3x3):
        //   A[0,0]=1, A[0,2]=3
        //   A[1,1]=2
        //   A[2,0]=1, A[2,2]=2
        //
        // Dense B:
        //   [[1, 2, 3],
        //    [4, 5, 6],
        //    [7, 8, 9]]
        //
        // C[i,k] = min_j( A[i,j] + B[j,k] )  (MIN_PLUS semiring)
        //
        // Row 0 (j=0 w=1, j=2 w=3):
        //   C[0,0] = min(1+1, 3+7) = min(2,10) = 2
        //   C[0,1] = min(1+2, 3+8) = min(3,11) = 3
        //   C[0,2] = min(1+3, 3+9) = min(4,12) = 4
        //
        // Row 1 (j=1 w=2):
        //   C[1,0] = 2+4 = 6
        //   C[1,1] = 2+5 = 7
        //   C[1,2] = 2+6 = 8
        //
        // Row 2 (j=0 w=1, j=2 w=2):
        //   C[2,0] = min(1+1, 2+7) = min(2, 9) = 2
        //   C[2,1] = min(1+2, 2+8) = min(3,10) = 3
        //   C[2,2] = min(1+3, 2+9) = min(4,11) = 4
        INDArray aValues = Nd4j.createFromArray(new float[]{1f, 3f,   2f,   1f, 2f});
        INDArray aColIdx = Nd4j.createFromArray(new int[]{0, 2,    1,    0, 2});
        INDArray aRowPtr = Nd4j.createFromArray(new int[]{0, 2, 3, 5});
        SparseNDArray A = new SparseNDArray(aValues, aColIdx, aRowPtr,
                new long[]{3, 3}, SparseFormat.CSR);

        INDArray B = Nd4j.createFromArray(new float[][]{
            {1f, 2f, 3f},
            {4f, 5f, 6f},
            {7f, 8f, 9f}
        });

        CsrSpmmSemiring op = new CsrSpmmSemiring(
                A.getValues(), A.getColIdx(), A.getRowPtr(), B,
                3L, 3L, Semiring.MIN_PLUS);
        INDArray C = Nd4j.exec(op)[0];

        INDArray expected = Nd4j.createFromArray(new float[][]{
            {2f, 3f, 4f},
            {6f, 7f, 8f},
            {2f, 3f, 4f}
        });
        assertClose(expected.reshape(9), C.reshape(9), TOL, "SpMM MIN_PLUS 3x3");
    }
}
