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

package org.nd4j.linalg.api.ops.impl.graph;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * GraphSAGE-style multi-hop neighbour sampling for minibatch GNN training on graphs too
 * large for full-batch execution (the "systems" piece that complements the {@code sd.gnn().*}
 * layers).
 *
 * <p>Given a graph in CSR form ({@code rowPtr}/{@code colIdx} where {@code colIdx} lists the
 * source/neighbour nodes of each row) and a set of <em>seed</em> nodes, {@link #sampleSubgraph}
 * expands {@code fanouts.length} hops outward, sampling up to {@code fanouts[hop]} neighbours
 * per frontier node, and returns the induced sub-graph re-indexed to dense local indices. The
 * returned CSR ({@link SampledSubgraph#rowPtr}/{@link SampledSubgraph#colIdx}) can be fed
 * directly to the sparse GNN ops (e.g. {@code CsrEdgeGather}/{@code CsrEdgeAggregate}); node
 * features for the batch are obtained by gathering rows {@link SampledSubgraph#nodeIds} from the
 * full feature matrix.
 *
 * <p>The seed nodes always occupy the first {@link SampledSubgraph#numSeeds} local indices, so
 * the model's per-seed outputs are {@code out[0 .. numSeeds)}.
 *
 * <p>Pure CPU/JVM logic over {@code int[]} adjacency — no native ops, no backend dependency.
 * Sampling is uniform without replacement and fully reproducible from {@code randomSeed}.
 */
public final class GraphSampler {

    private GraphSampler() {
        // static utility — not instantiable
    }

    /** Result of a multi-hop neighbourhood sample, re-indexed to dense local node indices. */
    public static final class SampledSubgraph {
        /** Global node id for each local index [numNodes]; the first {@link #numSeeds} are the seeds. */
        public final int[] nodeIds;
        /** Induced-CSR row pointers over local indices [numNodes+1]. */
        public final int[] rowPtr;
        /** Induced-CSR neighbour (source) local indices [numEdges]. */
        public final int[] colIdx;
        /** Number of seed nodes (the first {@code numSeeds} entries of {@link #nodeIds}). */
        public final int numSeeds;

        SampledSubgraph(int[] nodeIds, int[] rowPtr, int[] colIdx, int numSeeds) {
            this.nodeIds = nodeIds;
            this.rowPtr = rowPtr;
            this.colIdx = colIdx;
            this.numSeeds = numSeeds;
        }

        public int numNodes() { return nodeIds.length; }
        public int numEdges() { return colIdx.length; }

        /** Global node ids as an INT32 {@link INDArray} (use to gather batch features). */
        public INDArray nodeIdsArr() { return Nd4j.createFromArray(nodeIds); }
        /** Induced-CSR row pointers as an INT32 {@link INDArray}. */
        public INDArray rowPtrArr() { return Nd4j.createFromArray(rowPtr); }
        /** Induced-CSR neighbour indices as an INT32 {@link INDArray}. */
        public INDArray colIdxArr() { return Nd4j.createFromArray(colIdx); }
    }

    /**
     * Multi-hop GraphSAGE neighbour sample.
     *
     * @param rowPtr     full-graph CSR row pointers [N+1]
     * @param colIdx     full-graph CSR neighbour indices [nnz]
     * @param seeds      seed node global ids (duplicates are de-duplicated, order preserved)
     * @param fanouts    neighbours to sample per hop; {@code fanouts.length} = number of hops
     * @param randomSeed RNG seed for reproducible sampling
     * @return the induced sub-graph re-indexed to local node indices
     */
    public static SampledSubgraph sampleSubgraph(int[] rowPtr, int[] colIdx,
                                                 int[] seeds, int[] fanouts, long randomSeed) {
        if (rowPtr == null || rowPtr.length < 2) {
            throw new IllegalArgumentException("rowPtr must have length >= 2 (N+1)");
        }
        final int n = rowPtr.length - 1;
        final Random rng = new Random(randomSeed);

        // global node id -> local index, in discovery order (seeds first)
        LinkedHashMap<Integer, Integer> g2l = new LinkedHashMap<>();
        for (int s : seeds) {
            if (s < 0 || s >= n) {
                throw new IllegalArgumentException("seed " + s + " out of range [0," + n + ")");
            }
            g2l.putIfAbsent(s, g2l.size());
        }
        final int numSeeds = g2l.size();

        // edges as (localDst, localSrc): local node `dst` aggregates from neighbour `src`
        List<int[]> edges = new ArrayList<>();
        List<Integer> frontier = new ArrayList<>(g2l.keySet());

        for (int hop = 0; hop < fanouts.length && !frontier.isEmpty(); hop++) {
            final int fanout = fanouts[hop];
            List<Integer> next = new ArrayList<>();
            for (int gNode : frontier) {
                final int start = rowPtr[gNode];
                final int end = rowPtr[gNode + 1];
                final int deg = end - start;
                if (deg <= 0) continue;
                final int k = Math.min(fanout, deg);
                final int lDst = g2l.get(gNode);
                for (int off : sampleOffsets(start, end, k, rng)) {
                    final int gSrc = colIdx[off];
                    Integer lSrc = g2l.get(gSrc);
                    if (lSrc == null) {
                        lSrc = g2l.size();
                        g2l.put(gSrc, lSrc);
                        next.add(gSrc);
                    }
                    edges.add(new int[]{lDst, lSrc});
                }
            }
            frontier = next;
        }

        final int numNodes = g2l.size();
        final int[] nodeIds = new int[numNodes];
        for (Map.Entry<Integer, Integer> e : g2l.entrySet()) {
            nodeIds[e.getValue()] = e.getKey();
        }

        // build induced CSR: group edges by local dst (stable, ascending)
        edges.sort((a, b) -> a[0] != b[0] ? Integer.compare(a[0], b[0]) : Integer.compare(a[1], b[1]));
        final int[] newRowPtr = new int[numNodes + 1];
        final int[] newColIdx = new int[edges.size()];
        int ei = 0;
        for (int node = 0; node < numNodes; node++) {
            newRowPtr[node] = ei;
            while (ei < edges.size() && edges.get(ei)[0] == node) {
                newColIdx[ei] = edges.get(ei)[1];
                ei++;
            }
        }
        newRowPtr[numNodes] = ei;

        return new SampledSubgraph(nodeIds, newRowPtr, newColIdx, numSeeds);
    }

    /** Convenience overload taking INDArray CSR (INT32). */
    public static SampledSubgraph sampleSubgraph(INDArray rowPtr, INDArray colIdx,
                                                 int[] seeds, int[] fanouts, long randomSeed) {
        return sampleSubgraph(rowPtr.toIntVector(), colIdx.toIntVector(), seeds, fanouts, randomSeed);
    }

    /**
     * Partition {@code allSeeds} into fixed-size minibatches and sample each.  Useful as the
     * per-epoch loader for minibatch GNN training; each batch is sampled with a distinct,
     * reproducible RNG stream derived from {@code randomSeed}.
     *
     * @param rowPtr     full-graph CSR row pointers [N+1]
     * @param colIdx     full-graph CSR neighbour indices [nnz]
     * @param allSeeds   all training/target node ids to cover
     * @param batchSize  seeds per minibatch (last batch may be smaller)
     * @param fanouts    neighbours to sample per hop
     * @param randomSeed base RNG seed
     * @return one {@link SampledSubgraph} per minibatch, in order
     */
    public static List<SampledSubgraph> sampleMiniBatches(int[] rowPtr, int[] colIdx,
                                                          int[] allSeeds, int batchSize,
                                                          int[] fanouts, long randomSeed) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be > 0");
        }
        List<SampledSubgraph> batches = new ArrayList<>();
        for (int b = 0; b * batchSize < allSeeds.length; b++) {
            final int from = b * batchSize;
            final int to = Math.min(from + batchSize, allSeeds.length);
            final int[] batchSeeds = new int[to - from];
            System.arraycopy(allSeeds, from, batchSeeds, 0, to - from);
            // distinct, reproducible stream per batch
            batches.add(sampleSubgraph(rowPtr, colIdx, batchSeeds, fanouts, randomSeed * 1000003L + b));
        }
        return batches;
    }

    /**
     * Sample {@code k} distinct offsets uniformly from {@code [start, end)} without replacement.
     * When {@code k >= end-start} all offsets are returned (full neighbourhood).
     */
    private static int[] sampleOffsets(int start, int end, int k, Random rng) {
        final int deg = end - start;
        if (k >= deg) {
            int[] all = new int[deg];
            for (int i = 0; i < deg; i++) all[i] = start + i;
            return all;
        }
        // partial Fisher-Yates over a local copy of the offsets
        final int[] pool = new int[deg];
        for (int i = 0; i < deg; i++) pool[i] = start + i;
        final int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            final int j = i + rng.nextInt(deg - i);
            final int tmp = pool[i];
            pool[i] = pool[j];
            pool[j] = tmp;
            out[i] = pool[i];
        }
        return out;
    }
}
