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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * Deterministic negative edge sampler for link-prediction benchmarking.
 *
 * <p>Given the set of existing (positive) edges and the number of nodes, this class
 * samples edges that do NOT appear in the existing edge set.  Self-loops can optionally
 * be excluded.  The same seed always produces the same samples (deterministic).
 *
 * <p>Negative edges are encoded as {@code src * numNodes + dst} in a {@link HashSet} for O(1)
 * lookup; the positive-edge set is supplied at construction time and is never mutated.
 *
 * <p>Usage:
 * <pre>{@code
 *   // Build from an [E, 2] edge-list array:
 *   NegativeEdgeSampler sampler =
 *       NegativeEdgeSampler.fromEdgeList(edgeList, numNodes, /*allowSelfLoops=*\/false, 42L);
 *   INDArray negEdges = sampler.sample(200);  // [200, 2] array of non-existing edges
 * }</pre>
 */
public class NegativeEdgeSampler {

    private final long numNodes;
    private final Set<Long> existingEdges;
    private final boolean allowSelfLoops;
    private final long seed;

    /**
     * Construct a sampler with a pre-built existing-edge set.
     *
     * @param numNodes      total number of nodes in the graph (nodes are 0-indexed)
     * @param existingEdges set of existing edges encoded as {@code src * numNodes + dst}
     * @param allowSelfLoops if {@code false}, self-loops ({@code src == dst}) are never returned
     * @param seed          random seed for reproducibility
     */
    public NegativeEdgeSampler(long numNodes, Set<Long> existingEdges,
                                boolean allowSelfLoops, long seed) {
        if (numNodes < 2) throw new IllegalArgumentException("numNodes must be >= 2, got " + numNodes);
        this.numNodes      = numNodes;
        this.existingEdges = new HashSet<>(existingEdges); // defensive copy
        this.allowSelfLoops = allowSelfLoops;
        this.seed          = seed;
    }

    // -------------------------------------------------------------------------
    // Factory helpers
    // -------------------------------------------------------------------------

    /**
     * Build a sampler from an {@code [E, 2]} edge-list INDArray.
     *
     * @param edgeList      INDArray of shape {@code [E, 2]} with integer node indices
     * @param numNodes      total number of nodes
     * @param allowSelfLoops whether self-loops are valid negatives
     * @param seed          random seed
     */
    public static NegativeEdgeSampler fromEdgeList(INDArray edgeList, long numNodes,
                                                    boolean allowSelfLoops, long seed) {
        if (edgeList.rank() != 2 || edgeList.size(1) != 2) {
            throw new IllegalArgumentException(
                    "edgeList must have shape [E, 2], got " +
                            Arrays.toString(edgeList.shape()));
        }
        int E = (int) edgeList.size(0);
        Set<Long> existingEdges = new HashSet<>(E * 2);
        for (int i = 0; i < E; i++) {
            long src = edgeList.getLong(i, 0);
            long dst = edgeList.getLong(i, 1);
            existingEdges.add(encode(src, dst, numNodes));
        }
        return new NegativeEdgeSampler(numNodes, existingEdges, allowSelfLoops, seed);
    }

    /**
     * Build a sampler from a flat collection of (src, dst) pairs (for convenience in tests).
     *
     * @param edges     collection of long[2] arrays, each containing {src, dst}
     * @param numNodes  total number of nodes
     * @param allowSelfLoops whether self-loops are valid negatives
     * @param seed      random seed
     */
    public static NegativeEdgeSampler fromPairs(Iterable<long[]> edges, long numNodes,
                                                 boolean allowSelfLoops, long seed) {
        Set<Long> existingEdges = new HashSet<>();
        for (long[] e : edges) {
            existingEdges.add(encode(e[0], e[1], numNodes));
        }
        return new NegativeEdgeSampler(numNodes, existingEdges, allowSelfLoops, seed);
    }

    // -------------------------------------------------------------------------
    // Sampling
    // -------------------------------------------------------------------------

    /**
     * Sample {@code n} negative (non-existing) edges.
     *
     * <p>The sampler uses rejection sampling; for sparse graphs (where most candidate
     * pairs are not in the existing-edge set) this converges quickly.  For very dense
     * graphs, performance degrades — callers should ensure the graph is not saturated.
     *
     * @param n number of negative edges to sample; must be &gt; 0
     * @return INDArray of shape {@code [n, 2]} with integer node indices (LONG dtype)
     * @throws IllegalStateException if the graph is too dense to find {@code n} unique
     *         negatives (after a bounded number of rejections)
     */
    public INDArray sample(int n) {
        if (n <= 0) throw new IllegalArgumentException("n must be > 0, got " + n);

        // Maximum possible negative edges given self-loop policy.
        long maxPossible = allowSelfLoops
                ? numNodes * numNodes - existingEdges.size()
                : numNodes * (numNodes - 1) - existingEdges.size();
        // Also subtract self-loop count from existingEdges if they were present there
        // (users may have passed a set that includes self-loops).
        if (maxPossible < n) {
            throw new IllegalStateException(
                    "Cannot sample " + n + " unique negative edges from a graph with " +
                            numNodes + " nodes and " + existingEdges.size() + " existing edges" +
                            (allowSelfLoops ? "" : " (self-loops excluded)") +
                            ". Maximum possible negatives: " + maxPossible);
        }

        Random rng = new Random(seed);
        Set<Long> sampled = new HashSet<>(n * 2);
        long[] data = new long[n * 2];
        int count = 0;
        int maxRetries = Math.max(n * 100, 1_000_000);
        int tries = 0;

        while (count < n) {
            if (tries++ > maxRetries) {
                throw new IllegalStateException(
                        "Exceeded retry limit (" + maxRetries + ") when sampling negative edges. " +
                                "The graph may be too dense.");
            }
            long src = (long) (rng.nextDouble() * numNodes);
            long dst = (long) (rng.nextDouble() * numNodes);

            // Apply self-loop filter.
            if (!allowSelfLoops && src == dst) continue;

            long code = encode(src, dst, numNodes);

            // Skip if this edge already exists (positive or already sampled negative).
            if (existingEdges.contains(code)) continue;
            if (!sampled.add(code)) continue; // duplicate sample

            data[count * 2]     = src;
            data[count * 2 + 1] = dst;
            count++;
        }

        return Nd4j.createFromArray(data).reshape(n, 2);
    }

    // -------------------------------------------------------------------------
    // Accessors for testing / introspection
    // -------------------------------------------------------------------------

    /** Number of nodes used by this sampler. */
    public long getNumNodes() { return numNodes; }

    /** Read-only view of the existing-edge set (encoded). */
    public Set<Long> getExistingEdges() { return Collections.unmodifiableSet(existingEdges); }

    /**
     * Check whether the given directed edge is in the existing (positive) edge set.
     *
     * @param src source node index
     * @param dst destination node index
     */
    public boolean isExistingEdge(long src, long dst) {
        return existingEdges.contains(encode(src, dst, numNodes));
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /** Encode a directed edge as a single long for O(1) set membership. */
    static long encode(long src, long dst, long numNodes) {
        return src * numNodes + dst;
    }
}
