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

import java.util.*;

/**
 * Greedy Louvain community detection (modularity maximisation).
 *
 * <p>Implements the two-phase Louvain algorithm:
 * <ul>
 *   <li><b>Phase 1</b>: Local modularity optimisation — for each node, try moving
 *       it to the community of each neighbour and commit whichever move gives the
 *       largest positive delta-Q. Repeat until no node moves improve modularity.</li>
 *   <li><b>Phase 2</b>: Network aggregation — collapse each community into a
 *       single "super-node", building a weighted meta-graph; then repeat phase 1
 *       on the meta-graph.</li>
 * </ul>
 *
 * <p>The two phases alternate until no further improvement is possible.
 *
 * <p>Modularity is defined as:
 * <pre>
 *   Q = (1 / 2m) * sum_{i,j} [ A_{ij} - k_i*k_j/(2m) ] * delta(c_i, c_j)
 * </pre>
 * where m = total edge weight, k_i = weighted degree of i, c_i = community of i.
 *
 * <p>The algorithm is deterministic given the node-visit order (currently sequential
 * 0..n-1 in each pass).
 *
 * <p>Input: symmetric, non-negative adjacency matrix. Self-loops are allowed but
 * do not affect community assignment.
 */
public class LouvainCommunityDetection {

    private LouvainCommunityDetection() {}

    /**
     * Result returned by {@link #detect(INDArray)}.
     */
    public static class Result {
        /** Community label for each node (0-based, arbitrary but consistent). */
        public final int[] communities;
        /** Final modularity Q (higher is better; >0 means better-than-random). */
        public final double modularity;

        public Result(int[] communities, double modularity) {
            this.communities = communities;
            this.modularity = modularity;
        }
    }

    /**
     * Run Louvain community detection on an adjacency matrix.
     *
     * @param adjacency symmetric non-negative adjacency matrix, shape [n, n]
     * @return {@link Result} with per-node community assignments and final modularity
     */
    public static Result detect(INDArray adjacency) {
        Preconditions.checkArgument(adjacency.rank() == 2 && adjacency.rows() == adjacency.columns(),
                "adjacency must be a square matrix; got shape %s", Arrays.toString(adjacency.shape()));

        int n = (int) adjacency.rows();

        // Extract adjacency as double[][] for fast access
        double[][] adj = toJava(adjacency.castTo(DataType.DOUBLE), n);

        // Top-level partition (maps original node -> final community index)
        int[] globalCommunity = runLouvain(adj, n);

        // Relabel communities 0..C-1
        globalCommunity = relabel(globalCommunity, n);

        double q = computeModularity(adj, n, globalCommunity);
        return new Result(globalCommunity, q);
    }

    // -------------------------------------------------------------------------
    // Core algorithm
    // -------------------------------------------------------------------------

    /**
     * Full Louvain: repeatedly apply phase-1 + phase-2 until stable.
     * Returns community labels in the original node space.
     */
    private static int[] runLouvain(double[][] adj, int n) {
        // originalNode[i] = which original nodes are compressed into super-node i
        // We track this as a partition: partition[originalNode] = superNode
        int[] partition = new int[n];
        for (int i = 0; i < n; i++) partition[i] = i;

        double[][] currentAdj = adj;
        int currentN = n;

        while (true) {
            // Phase 1: local modularity optimisation
            int[] localComm = phase1(currentAdj, currentN);

            // Check if anything changed (all nodes still in their own singleton)
            boolean anyMoved = false;
            for (int i = 0; i < currentN; i++) {
                if (localComm[i] != i) {
                    anyMoved = true;
                    break;
                }
            }

            // Map local communities back through the running partition
            for (int orig = 0; orig < n; orig++) {
                partition[orig] = localComm[partition[orig]];
            }

            if (!anyMoved) break; // converged

            // Phase 2: build meta-graph (super-nodes = communities)
            int[] localRelabeled = relabel(localComm, currentN);
            int numComm = maxVal(localRelabeled, currentN) + 1;

            double[][] metaAdj = buildMetaGraph(currentAdj, currentN, localRelabeled, numComm);

            currentAdj = metaAdj;
            currentN = numComm;

            // Remap partition through relabeling
            for (int orig = 0; orig < n; orig++) {
                partition[orig] = localRelabeled[partition[orig]];
            }
        }

        return partition;
    }

    /**
     * Phase 1: greedy local moves.
     * Returns local community label per node (may equal node index if not moved).
     */
    private static int[] phase1(double[][] adj, int n) {
        // Compute weighted degrees and total edge weight
        double[] degree = new double[n];
        double totalWeight = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                degree[i] += adj[i][j];
            }
            totalWeight += degree[i];
        }
        totalWeight /= 2.0; // each edge counted twice
        if (totalWeight == 0) {
            int[] singletons = new int[n];
            for (int i = 0; i < n; i++) singletons[i] = i;
            return singletons;
        }

        // community[i] = current community of node i (init: each in own community)
        int[] community = new int[n];
        for (int i = 0; i < n; i++) community[i] = i;

        // communityTotalDeg[c] = sum of degrees of nodes in community c
        // communityInternalWeight[c] = sum of weights inside community c (self-loops count once)
        double[] commDeg = new double[n]; // indexed by original community labels
        double[] commIn = new double[n];
        for (int i = 0; i < n; i++) {
            commDeg[i] = degree[i];
            commIn[i] = adj[i][i]; // self-loops
        }

        boolean improved = true;
        while (improved) {
            improved = false;
            for (int i = 0; i < n; i++) {
                int ci = community[i];

                // Compute k_{i,ci}: weight of edges from i to nodes in community ci (excluding i)
                double kInCi = weightToComm(adj[i], community, ci, n) - adj[i][i];

                // deltaQ of removing i from ci
                double sigma_tot_ci = commDeg[ci] - degree[i];
                double dq_remove = kInCi - degree[i] * sigma_tot_ci / (2.0 * totalWeight);

                // Find best neighbouring community to move i to
                // Gather unique neighbouring communities
                Set<Integer> neighborComms = new LinkedHashSet<>();
                for (int j = 0; j < n; j++) {
                    if (j != i && adj[i][j] > 0) {
                        neighborComms.add(community[j]);
                    }
                }

                int bestComm = ci;
                double bestDeltaQ = 0; // only move if strictly positive gain

                for (int cj : neighborComms) {
                    if (cj == ci) continue;
                    double kInCj = weightToComm(adj[i], community, cj, n);
                    double sigma_tot_cj = commDeg[cj];
                    double dq_add = kInCj - degree[i] * sigma_tot_cj / (2.0 * totalWeight);
                    double deltaQ = dq_add - dq_remove;
                    if (deltaQ > bestDeltaQ + 1e-15) {
                        bestDeltaQ = deltaQ;
                        bestComm = cj;
                    }
                }

                if (bestComm != ci) {
                    // Move i from ci to bestComm
                    commDeg[ci] -= degree[i];
                    commIn[ci] -= kInCi + adj[i][i];

                    community[i] = bestComm;
                    commDeg[bestComm] += degree[i];
                    double kInBest = weightToComm(adj[i], community, bestComm, n) - adj[i][i];
                    commIn[bestComm] += kInBest + adj[i][i];

                    improved = true;
                }
            }
        }

        return community;
    }

    /** Sum of adj[i][j] for all j in community c. */
    private static double weightToComm(double[] rowI, int[] community, int c, int n) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            if (community[j] == c) sum += rowI[j];
        }
        return sum;
    }

    /** Build a weighted meta-graph by collapsing communities into super-nodes. */
    private static double[][] buildMetaGraph(double[][] adj, int n, int[] community, int numComm) {
        double[][] meta = new double[numComm][numComm];
        for (int i = 0; i < n; i++) {
            int ci = community[i];
            for (int j = 0; j < n; j++) {
                if (adj[i][j] > 0) {
                    int cj = community[j];
                    meta[ci][cj] += adj[i][j];
                }
            }
        }
        // Self-loops are doubled during aggregation (each internal edge i->j, j->i both contribute)
        // We keep the meta-graph as-is (symmetric, self-loops allowed)
        return meta;
    }

    // -------------------------------------------------------------------------
    // Modularity
    // -------------------------------------------------------------------------

    /**
     * Compute modularity Q for the given community assignment.
     */
    public static double computeModularity(double[][] adj, int n, int[] community) {
        double totalWeight = 0;
        double[] degree = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                degree[i] += adj[i][j];
                totalWeight += adj[i][j];
            }
        }
        totalWeight /= 2.0;
        if (totalWeight == 0) return 0;

        double q = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (community[i] == community[j]) {
                    q += adj[i][j] - degree[i] * degree[j] / (2.0 * totalWeight);
                }
            }
        }
        return q / (2.0 * totalWeight);
    }

    // -------------------------------------------------------------------------
    // Utilities
    // -------------------------------------------------------------------------

    private static double[][] toJava(INDArray a, int n) {
        double[][] out = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                out[i][j] = a.getDouble(i, j);
            }
        }
        return out;
    }

    /** Relabel community indices to 0..C-1 (compact, preserving relative order). */
    static int[] relabel(int[] comm, int n) {
        Map<Integer, Integer> map = new LinkedHashMap<>();
        int next = 0;
        for (int i = 0; i < n; i++) {
            if (!map.containsKey(comm[i])) {
                map.put(comm[i], next++);
            }
        }
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = map.get(comm[i]);
        return out;
    }

    private static int maxVal(int[] arr, int n) {
        int max = 0;
        for (int i = 0; i < n; i++) if (arr[i] > max) max = arr[i];
        return max;
    }
}
