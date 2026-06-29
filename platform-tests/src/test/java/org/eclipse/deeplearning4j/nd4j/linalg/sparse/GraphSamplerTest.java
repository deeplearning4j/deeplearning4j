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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.graph.GraphSampler;
import org.nd4j.linalg.api.ops.impl.graph.GraphSampler.SampledSubgraph;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Correctness tests for {@link GraphSampler} (GraphSAGE-style multi-hop neighbour sampling).
 *
 * <h3>Test graph (N=6, undirected, 16 directed edges)</h3>
 * <pre>
 *   0: {1,2,3}   1: {0,2}     2: {0,1,3,4}
 *   3: {0,2,5}   4: {2,5}     5: {3,4}
 *   rowPtr = [0, 3, 5, 9, 12, 14, 16]
 *   colIdx = [1,2,3, 0,2, 0,1,3,4, 0,2,5, 2,5, 3,4]
 * </pre>
 */
@Tag(TagNames.SAMEDIFF)
public class GraphSamplerTest extends BaseNd4jTestWithBackends {

    private static final int[] ROW_PTR = {0, 3, 5, 9, 12, 14, 16};
    private static final int[] COL_IDX = {1, 2, 3, 0, 2, 0, 1, 3, 4, 0, 2, 5, 2, 5, 3, 4};

    /** Is {@code v} a neighbour (source) of {@code u} in the full graph? */
    private static boolean isNeighbor(int u, int v) {
        for (int k = ROW_PTR[u]; k < ROW_PTR[u + 1]; k++) {
            if (COL_IDX[k] == v) return true;
        }
        return false;
    }

    private static void assertValidSubgraph(SampledSubgraph g, int fanoutCap) {
        // rowPtr well-formed
        assertEquals(g.numNodes() + 1, g.rowPtr.length, "rowPtr length");
        assertEquals(0, g.rowPtr[0], "rowPtr[0]");
        assertEquals(g.colIdx.length, g.rowPtr[g.numNodes()], "rowPtr tail == numEdges");
        for (int i = 0; i < g.numNodes(); i++) {
            assertTrue(g.rowPtr[i] <= g.rowPtr[i + 1], "rowPtr monotonic at " + i);
            int deg = g.rowPtr[i + 1] - g.rowPtr[i];
            assertTrue(deg <= fanoutCap, "node " + i + " sampled deg " + deg + " > fanout " + fanoutCap);
        }
        // every local index maps to a valid global id; every sampled edge is a real edge
        for (int dst = 0; dst < g.numNodes(); dst++) {
            for (int k = g.rowPtr[dst]; k < g.rowPtr[dst + 1]; k++) {
                int lsrc = g.colIdx[k];
                assertTrue(lsrc >= 0 && lsrc < g.numNodes(), "local src in range");
                assertTrue(isNeighbor(g.nodeIds[dst], g.nodeIds[lsrc]),
                        "sampled edge (" + g.nodeIds[dst] + "<-" + g.nodeIds[lsrc] + ") is a real edge");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSeedsFirstAndValid(Nd4jBackend backend) {
        int[] seeds = {0, 3};
        SampledSubgraph g = GraphSampler.sampleSubgraph(ROW_PTR, COL_IDX, seeds, new int[]{2}, 42L);
        assertEquals(2, g.numSeeds, "numSeeds");
        assertEquals(0, g.nodeIds[0], "seed 0 first");
        assertEquals(3, g.nodeIds[1], "seed 3 second");
        assertValidSubgraph(g, 2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFanoutRespected(Nd4jBackend backend) {
        // every node has degree >= 2, fanout 1 → exactly 1 sampled neighbour per expanded node
        SampledSubgraph g = GraphSampler.sampleSubgraph(ROW_PTR, COL_IDX, new int[]{2}, new int[]{1, 1}, 7L);
        assertValidSubgraph(g, 1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFullFanoutGetsAllNeighbors(Nd4jBackend backend) {
        // fanout >= max degree (4) → seed keeps ALL its neighbours
        SampledSubgraph g = GraphSampler.sampleSubgraph(ROW_PTR, COL_IDX, new int[]{2}, new int[]{10}, 1L);
        // seed is node 2 (deg 4); its sampled row must contain all 4 neighbours {0,1,3,4}
        int seedLocal = 0; // node 2 is the only seed → local 0
        int deg = g.rowPtr[seedLocal + 1] - g.rowPtr[seedLocal];
        assertEquals(4, deg, "full fanout keeps all neighbours of node 2");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDeterministic(Nd4jBackend backend) {
        int[] seeds = {0, 5};
        int[] fan = {2, 2};
        SampledSubgraph a = GraphSampler.sampleSubgraph(ROW_PTR, COL_IDX, seeds, fan, 123L);
        SampledSubgraph b = GraphSampler.sampleSubgraph(ROW_PTR, COL_IDX, seeds, fan, 123L);
        assertArrayEquals(a.nodeIds, b.nodeIds, "same seed → same nodeIds");
        assertArrayEquals(a.rowPtr, b.rowPtr, "same seed → same rowPtr");
        assertArrayEquals(a.colIdx, b.colIdx, "same seed → same colIdx");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiHopGrowsNeighborhood(Nd4jBackend backend) {
        SampledSubgraph oneHop = GraphSampler.sampleSubgraph(ROW_PTR, COL_IDX, new int[]{0}, new int[]{2}, 5L);
        SampledSubgraph twoHop = GraphSampler.sampleSubgraph(ROW_PTR, COL_IDX, new int[]{0}, new int[]{2, 2}, 5L);
        assertTrue(twoHop.numNodes() >= oneHop.numNodes(), "2-hop covers at least as many nodes");
        assertValidSubgraph(twoHop, 2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDedupSeeds(Nd4jBackend backend) {
        SampledSubgraph g = GraphSampler.sampleSubgraph(ROW_PTR, COL_IDX, new int[]{1, 1, 4}, new int[]{2}, 9L);
        assertEquals(2, g.numSeeds, "duplicate seed de-duplicated");
        assertEquals(1, g.nodeIds[0]);
        assertEquals(4, g.nodeIds[1]);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMiniBatchesCoverAllSeeds(Nd4jBackend backend) {
        int[] allSeeds = {0, 1, 2, 3, 4, 5};
        List<SampledSubgraph> batches =
                GraphSampler.sampleMiniBatches(ROW_PTR, COL_IDX, allSeeds, 2, new int[]{2}, 11L);
        assertEquals(3, batches.size(), "6 seeds / batch 2 → 3 batches");
        int totalSeeds = 0;
        for (SampledSubgraph g : batches) {
            totalSeeds += g.numSeeds;
            assertValidSubgraph(g, 2);
        }
        assertEquals(6, totalSeeds, "all seeds covered across batches");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIndArrayOverloadAndAccessors(Nd4jBackend backend) {
        INDArray rowPtr = Nd4j.createFromArray(ROW_PTR);
        INDArray colIdx = Nd4j.createFromArray(COL_IDX);
        SampledSubgraph g = GraphSampler.sampleSubgraph(rowPtr, colIdx, new int[]{0, 3}, new int[]{2, 2}, 99L);
        assertArrayEquals(g.nodeIds, g.nodeIdsArr().toIntVector(), "nodeIds INDArray matches");
        assertArrayEquals(g.rowPtr, g.rowPtrArr().toIntVector(), "rowPtr INDArray matches");
        assertArrayEquals(g.colIdx, g.colIdxArr().toIntVector(), "colIdx INDArray matches");
        assertValidSubgraph(g, 2);
    }
}
