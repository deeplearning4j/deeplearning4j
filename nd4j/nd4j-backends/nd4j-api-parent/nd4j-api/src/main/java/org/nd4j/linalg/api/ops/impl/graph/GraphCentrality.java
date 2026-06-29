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
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.graph;

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Per-node graph centrality measures for undirected graphs.
 *
 * <p>All methods accept a symmetric adjacency matrix (INDArray, any numeric type)
 * and return an INDArray of shape [n] (DOUBLE) with one centrality value per node.
 *
 * <h2>Closeness Centrality</h2>
 * <p>For node v:
 * <pre>
 *   closeness(v) = (R_v - 1) / sum_{u reachable, u != v} dist(v, u)
 * </pre>
 * where R_v is the number of nodes reachable from v (including v itself only if
 * R_v > 1). Isolated nodes receive closeness 0.
 * <p>Uses BFS for unweighted graphs (any non-zero adjacency entry = edge of weight 1).
 *
 * <h2>Betweenness Centrality</h2>
 * <p>Exact computation via <em>Brandes' algorithm</em> (Brandes, 2001).
 * <pre>
 *   betweenness(v) = sum_{s != v != t} sigma(s,t|v) / sigma(s,t)
 * </pre>
 * where sigma(s,t) = number of shortest paths from s to t, and sigma(s,t|v) =
 * those paths that pass through v.
 * <p>Uses BFS (unweighted graph: edge weight = 1 for any non-zero entry).
 * For undirected graphs, each ordered pair (s,t) is counted once
 * (the standard undirected convention — raw values are NOT divided by 2 from
 * a directed run; instead the BFS is run from each source and the unordered-pair
 * count is recovered by dividing by 2 at the end).
 * <p>Complexity: O(n * (n + m)) where m = number of edges.
 *
 * <p>No normalisation is applied by default. To normalise to [0,1], divide by
 * (n-1)(n-2)/2  (undirected) or  (n-1)(n-2)  (directed convention).
 */
public class GraphCentrality {

    private GraphCentrality() {}

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Compute closeness centrality for every node.
     *
     * @param adjacency symmetric adjacency matrix, shape [n, n]
     * @return INDArray of shape [n] (DOUBLE) — closeness per node
     */
    public static INDArray closeness(INDArray adjacency) {
        Preconditions.checkArgument(adjacency.rank() == 2 && adjacency.rows() == adjacency.columns(),
                "adjacency must be square; got shape %s", Arrays.toString(adjacency.shape()));
        int n = (int) adjacency.rows();
        boolean[][] neighbors = buildAdjList(adjacency, n);

        double[] result = new double[n];
        for (int s = 0; s < n; s++) {
            int[] dist = bfs(s, n, neighbors);
            long reachable = 0;
            long sumDist = 0;
            for (int t = 0; t < n; t++) {
                if (dist[t] >= 0 && t != s) {
                    reachable++;
                    sumDist += dist[t];
                }
            }
            if (sumDist > 0) {
                result[s] = (double) reachable / (double) sumDist;
            }
        }
        return Nd4j.create(result, new long[]{n}, DataType.DOUBLE);
    }

    /**
     * Compute betweenness centrality for every node via Brandes' algorithm.
     *
     * @param adjacency symmetric adjacency matrix, shape [n, n]
     * @return INDArray of shape [n] (DOUBLE) — raw (unnormalised) betweenness per node
     */
    public static INDArray betweenness(INDArray adjacency) {
        Preconditions.checkArgument(adjacency.rank() == 2 && adjacency.rows() == adjacency.columns(),
                "adjacency must be square; got shape %s", Arrays.toString(adjacency.shape()));
        int n = (int) adjacency.rows();
        boolean[][] neighbors = buildAdjList(adjacency, n);

        double[] cb = new double[n]; // cumulative betweenness

        for (int s = 0; s < n; s++) {
            // BFS from source s
            ArrayDeque<Integer> Q = new ArrayDeque<>();
            Deque<Integer> S = new ArrayDeque<>(); // stack of visited nodes, in BFS order
            @SuppressWarnings("unchecked")
            List<Integer>[] pred = new List[n]; // predecessors on shortest paths
            for (int i = 0; i < n; i++) pred[i] = new ArrayList<>();

            long[] sigma = new long[n];   // number of shortest paths from s
            int[] dist = new int[n];
            Arrays.fill(dist, -1);

            dist[s] = 0;
            sigma[s] = 1;
            Q.add(s);

            while (!Q.isEmpty()) {
                int v = Q.poll();
                S.push(v);
                for (int w = 0; w < n; w++) {
                    if (!neighbors[v][w]) continue;
                    // First visit to w?
                    if (dist[w] < 0) {
                        Q.add(w);
                        dist[w] = dist[v] + 1;
                    }
                    // Is (v, w) on a shortest path?
                    if (dist[w] == dist[v] + 1) {
                        sigma[w] += sigma[v];
                        pred[w].add(v);
                    }
                }
            }

            // Accumulate dependencies
            double[] delta = new double[n];
            while (!S.isEmpty()) {
                int w = S.pop();
                for (int v : pred[w]) {
                    delta[v] += ((double) sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
                if (w != s) {
                    cb[w] += delta[w];
                }
            }
        }

        // For undirected graphs: each pair (s,t) was counted twice (once from s, once from t)
        for (int i = 0; i < n; i++) {
            cb[i] /= 2.0;
        }

        return Nd4j.create(cb, new long[]{n}, DataType.DOUBLE);
    }

    // -------------------------------------------------------------------------
    // Internals
    // -------------------------------------------------------------------------

    /**
     * Build an adjacency boolean matrix: neighbors[i][j] = true iff adjacency[i][j] != 0.
     */
    private static boolean[][] buildAdjList(INDArray adjacency, int n) {
        INDArray a = adjacency.castTo(DataType.DOUBLE);
        boolean[][] nbr = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (a.getDouble(i, j) != 0.0) {
                    nbr[i][j] = true;
                }
            }
        }
        return nbr;
    }

    /**
     * BFS from source {@code s}. Returns dist[t] = shortest-path distance, or -1 if unreachable.
     */
    private static int[] bfs(int s, int n, boolean[][] neighbors) {
        int[] dist = new int[n];
        Arrays.fill(dist, -1);
        dist[s] = 0;
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        queue.add(s);
        while (!queue.isEmpty()) {
            int v = queue.poll();
            for (int w = 0; w < n; w++) {
                if (neighbors[v][w] && dist[w] < 0) {
                    dist[w] = dist[v] + 1;
                    queue.add(w);
                }
            }
        }
        return dist;
    }
}
