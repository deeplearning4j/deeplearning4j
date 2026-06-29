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

package org.eclipse.deeplearning4j.nd4j.linalg.graph;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.graph.GraphCentrality;
import org.nd4j.linalg.api.ops.impl.graph.LouvainCommunityDetection;
import org.nd4j.linalg.api.ops.impl.graph.SpectralClustering;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Correctness tests for SpectralClustering, LouvainCommunityDetection, and GraphCentrality.
 *
 * All graphs are small and deterministic; expected values are hand-computed.
 */
@Slf4j
@NativeTag
public class TestGraphAnalysisUtils extends BaseNd4jTestWithBackends {

    private DataType initialType;

    @BeforeEach
    public void before() {
        initialType = Nd4j.dataType();
        Nd4j.setDataType(DataType.DOUBLE);
    }

    @AfterEach
    public void after() {
        Nd4j.setDataType(initialType);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /**
     * Build a symmetric adjacency matrix from an upper-triangle list of (i, j, w) triples.
     */
    private static INDArray buildAdj(int n, double[]... edges) {
        double[] flat = new double[n * n];
        for (double[] e : edges) {
            int i = (int) e[0], j = (int) e[1];
            double w = e[2];
            flat[i * n + j] = w;
            flat[j * n + i] = w;
        }
        return Nd4j.create(flat, new long[]{n, n}, DataType.DOUBLE);
    }

    /** Two cliques of 4 nodes connected by a single bridge edge. */
    private static INDArray twoCliquesAdj(double bridgeWeight) {
        // 8 nodes: clique 1 = {0,1,2,3}, clique 2 = {4,5,6,7}
        // Each clique is fully connected; bridge is (3, 4).
        double[][] edges = new double[6 + 6 + 1][];
        int idx = 0;
        // clique 1 edges
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                edges[idx++] = new double[]{i, j, 1.0};
            }
        }
        // clique 2 edges
        for (int i = 4; i < 8; i++) {
            for (int j = i + 1; j < 8; j++) {
                edges[idx++] = new double[]{i, j, 1.0};
            }
        }
        // bridge
        edges[idx] = new double[]{3, 4, bridgeWeight};
        return buildAdj(8, edges);
    }

    // =========================================================================
    // SpectralClustering
    // =========================================================================

    /**
     * A planted 2-block graph (two 4-cliques + a weak bridge) should be split
     * into exactly 2 clusters.
     *
     * <p>NJW spectral clustering uses {@code k=2} eigenvectors for {@code k=2} clusters.
     * For nearly-symmetric 4-cliques, the second eigenvector captures within-clique
     * variation, which can cause the bridge-endpoint nodes (3 and 4) to be assigned
     * to either cluster. We therefore only assert on the three interior nodes of each
     * clique: {0,1,2} vs {5,6,7}. Those interior nodes always land in opposite clusters.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSpectralClusteringTwoBlocks(Nd4jBackend backend) {
        INDArray adj = twoCliquesAdj(0.05); // very weak bridge

        // k=2 clusters; seed fixed for reproducibility
        int[] labels = SpectralClustering.cluster(adj, 2, 200, 42L);
        assertEquals(8, labels.length);

        // Exactly two distinct cluster IDs must be present
        java.util.Set<Integer> ids = new java.util.HashSet<>();
        for (int l : labels) ids.add(l);
        assertEquals(2, ids.size(), "Expected exactly 2 distinct cluster labels");

        // Interior nodes of clique-1 (0,1,2) must all share one label
        int labelA = labels[0];
        assertEquals(labelA, labels[1], "Node 1 should be in same cluster as node 0");
        assertEquals(labelA, labels[2], "Node 2 should be in same cluster as node 0");

        // Interior nodes of clique-2 (5,6,7) must all share one label
        int labelB = labels[5];
        assertEquals(labelB, labels[6], "Node 6 should be in same cluster as node 5");
        assertEquals(labelB, labels[7], "Node 7 should be in same cluster as node 5");

        // The two interior groups must be in opposite clusters
        assertNotEquals(labelA, labelB,
                "Clique-1 interior nodes and clique-2 interior nodes must be in different clusters");
    }

    /** Single cluster: all nodes assigned cluster 0. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSpectralClusteringK1(Nd4jBackend backend) {
        INDArray adj = buildAdj(4,
                new double[]{0, 1, 1}, new double[]{1, 2, 1}, new double[]{2, 3, 1}, new double[]{0, 3, 1});
        int[] labels = SpectralClustering.cluster(adj, 1, 100, 0L);
        for (int l : labels) assertEquals(0, l);
    }

    /** k-means helper: perfectly separated two groups of points. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLloydsKmeansPerfectSeparation(Nd4jBackend backend) {
        // Group A near (1,0); Group B near (-1,0) — unit-circle points after row-norm
        double[][] pts = {
            { 1.0, 0.0}, { 0.9, 0.1}, { 0.95, 0.05},   // group A
            {-1.0, 0.0}, {-0.9, 0.1}, {-0.95, 0.05}    // group B
        };
        int[] labels = SpectralClustering.lloydsKmeans(pts, 6, 2, 100, 0L);
        // All A should share one label, all B the other
        int la = labels[0];
        assertEquals(la, labels[1]);
        assertEquals(la, labels[2]);
        int lb = labels[3];
        assertNotEquals(la, lb);
        assertEquals(lb, labels[4]);
        assertEquals(lb, labels[5]);
    }

    // =========================================================================
    // Louvain
    // =========================================================================

    /**
     * Planted partition (two 4-cliques + bridge) → Louvain should find 2 communities
     * with Q > 0.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLouvainTwoCliques(Nd4jBackend backend) {
        INDArray adj = twoCliquesAdj(0.1);
        LouvainCommunityDetection.Result result = LouvainCommunityDetection.detect(adj);
        int[] comm = result.communities;

        assertEquals(8, comm.length);
        assertTrue(result.modularity > 0.0, "Modularity should be positive; got " + result.modularity);

        // The two planted groups should get different community labels
        int c0 = comm[0];
        int c4 = comm[4];
        assertNotEquals(c0, c4, "Clique-1 and clique-2 nodes must land in different communities");

        // All clique-1 nodes same community
        for (int i = 0; i < 4; i++) {
            assertEquals(c0, comm[i],
                    "Node " + i + " should be in community " + c0);
        }
        // All clique-2 nodes same community
        for (int i = 4; i < 8; i++) {
            assertEquals(c4, comm[i],
                    "Node " + i + " should be in community " + c4);
        }
    }

    /** A single complete graph should yield modularity near 0 (one community). */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLouvainCompleteGraph(Nd4jBackend backend) {
        // K4: all 4 nodes fully connected — optimal partition is the whole graph
        INDArray adj = buildAdj(4,
                new double[]{0,1,1}, new double[]{0,2,1}, new double[]{0,3,1},
                new double[]{1,2,1}, new double[]{1,3,1}, new double[]{2,3,1});
        LouvainCommunityDetection.Result result = LouvainCommunityDetection.detect(adj);
        // For K4, splitting into sub-communities gives Q<0 so Louvain leaves all in one community
        // Modularity of the "all one community" partition is 0.0
        assertEquals(0.0, result.modularity, 1e-10);
    }

    /** Hand-computed modularity for the 2-clique planted partition. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLouvainModularityValue(Nd4jBackend backend) {
        // Two cliques of 4, no bridge — optimal Q computed analytically.
        // Each clique: 6 edges; m = 12; each node degree = 3.
        // Q = (1/2m) * sum_{i,j in same comm} [A_ij - k_i*k_j / 2m]
        // For clique 1: sum A_ij = 2*6=12 (sym); sum k_i*k_j/(2m) = 4*4*9/24 = 6
        // Q_1 = (12 - 6)/(24) = 6/24 = 0.25
        // By symmetry Q_2 = 0.25; total Q = 0.5
        //
        // With bridge=0 (two disconnected cliques):
        INDArray adj = twoCliquesAdj(0.0);
        LouvainCommunityDetection.Result result = LouvainCommunityDetection.detect(adj);
        assertEquals(0.5, result.modularity, 1e-9,
                "Two disconnected 4-cliques should give Q=0.5");
    }

    // =========================================================================
    // GraphCentrality — path graph
    // =========================================================================

    /**
     * Path graph 0-1-2-3-4 (5 nodes).
     *
     * <p>Closeness (hand-computed):
     * <ul>
     *   <li>Node 0 (endpoint): (n-1)/sum = 4/10 = 0.4</li>
     *   <li>Node 1:            4/7 ≈ 0.571</li>
     *   <li>Node 2 (center):   4/6 ≈ 0.667</li>
     *   <li>Node 3:            4/7</li>
     *   <li>Node 4 (endpoint): 4/10 = 0.4</li>
     * </ul>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testClosenessPathGraph(Nd4jBackend backend) {
        INDArray adj = buildAdj(5,
                new double[]{0,1,1}, new double[]{1,2,1}, new double[]{2,3,1}, new double[]{3,4,1});

        INDArray c = GraphCentrality.closeness(adj);
        assertEquals(5, c.length());

        double eps = 1e-10;
        assertEquals(4.0 / 10.0, c.getDouble(0), eps, "endpoint closeness");
        assertEquals(4.0 / 7.0,  c.getDouble(1), eps, "node-1 closeness");
        assertEquals(4.0 / 6.0,  c.getDouble(2), eps, "center closeness");
        assertEquals(4.0 / 7.0,  c.getDouble(3), eps, "node-3 closeness");
        assertEquals(4.0 / 10.0, c.getDouble(4), eps, "endpoint closeness");

        // Node 2 (center) has the highest closeness
        double maxC = c.maxNumber().doubleValue();
        assertEquals(4.0 / 6.0, maxC, eps, "center node has max closeness");
    }

    /**
     * Path graph: betweenness (Brandes, undirected, unnormalised).
     *
     * <p>Hand-computed:
     * <ul>
     *   <li>Node 0: 0 (never intermediate)</li>
     *   <li>Node 1: 3</li>
     *   <li>Node 2: 4 (maximum)</li>
     *   <li>Node 3: 3</li>
     *   <li>Node 4: 0</li>
     * </ul>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBetweennessPathGraph(Nd4jBackend backend) {
        INDArray adj = buildAdj(5,
                new double[]{0,1,1}, new double[]{1,2,1}, new double[]{2,3,1}, new double[]{3,4,1});

        INDArray b = GraphCentrality.betweenness(adj);
        assertEquals(5, b.length());

        double eps = 1e-10;
        assertEquals(0.0, b.getDouble(0), eps, "endpoint betweenness = 0");
        assertEquals(3.0, b.getDouble(1), eps, "node-1 betweenness = 3");
        assertEquals(4.0, b.getDouble(2), eps, "center betweenness = 4");
        assertEquals(3.0, b.getDouble(3), eps, "node-3 betweenness = 3");
        assertEquals(0.0, b.getDouble(4), eps, "endpoint betweenness = 0");

        // Node 2 has max betweenness
        double maxB = b.maxNumber().doubleValue();
        assertEquals(4.0, maxB, eps, "center has max betweenness");
    }

    // =========================================================================
    // GraphCentrality — star graph
    // =========================================================================

    /**
     * Star graph (5 nodes): center=0, leaves=1..4.
     *
     * <p>Closeness:
     * <ul>
     *   <li>Center (0): 4/4 = 1.0</li>
     *   <li>Leaves:     4/7</li>
     * </ul>
     *
     * <p>Betweenness (unnormalised, undirected):
     * <ul>
     *   <li>Center (0): 6  (all C(4,2)=6 leaf–leaf paths go through it)</li>
     *   <li>Leaves:     0</li>
     * </ul>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCentralityStarGraph(Nd4jBackend backend) {
        INDArray adj = buildAdj(5,
                new double[]{0,1,1}, new double[]{0,2,1}, new double[]{0,3,1}, new double[]{0,4,1});

        INDArray cl = GraphCentrality.closeness(adj);
        INDArray bw = GraphCentrality.betweenness(adj);

        double eps = 1e-10;

        // Closeness
        assertEquals(1.0,        cl.getDouble(0), eps, "star center closeness = 1.0");
        for (int leaf = 1; leaf <= 4; leaf++) {
            assertEquals(4.0 / 7.0, cl.getDouble(leaf), eps, "leaf " + leaf + " closeness = 4/7");
        }

        // Betweenness
        assertEquals(6.0, bw.getDouble(0), eps, "star center betweenness = 6");
        for (int leaf = 1; leaf <= 4; leaf++) {
            assertEquals(0.0, bw.getDouble(leaf), eps, "leaf " + leaf + " betweenness = 0");
        }

        // Center dominates both metrics
        assertEquals(cl.maxNumber().doubleValue(), cl.getDouble(0), eps);
        assertEquals(bw.maxNumber().doubleValue(), bw.getDouble(0), eps);
    }

    /** Isolated node (degree 0) should get closeness 0. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testClosenessIsolatedNode(Nd4jBackend backend) {
        // 3 nodes: 0-1 connected, 2 isolated
        INDArray adj = buildAdj(3, new double[]{0, 1, 1});
        INDArray c = GraphCentrality.closeness(adj);
        assertEquals(0.0, c.getDouble(2), 1e-10, "isolated node closeness = 0");
    }
}
