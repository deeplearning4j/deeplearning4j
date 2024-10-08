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

package org.deeplearning4j.graph.data;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.graph.api.Edge;
import org.deeplearning4j.graph.api.IGraph;
import org.deeplearning4j.graph.data.impl.DelimitedEdgeLineProcessor;
import org.deeplearning4j.graph.data.impl.DelimitedVertexLoader;
import org.deeplearning4j.graph.Graph;
import org.deeplearning4j.graph.vertexfactory.StringVertexFactory;
import org.deeplearning4j.graph.vertexfactory.VertexFactory;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.tests.tags.TagNames;

import java.io.IOException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@Disabled("Permissions issues on CI")
@Tag(TagNames.JAVA_ONLY)
@Tag(TagNames.FILE_IO)
public class TestGraphLoading extends BaseDL4JTest {

    @Test()
    @Timeout(10000)
    public void testEdgeListGraphLoading() throws IOException {
        ClassPathResource cpr = new ClassPathResource("deeplearning4j-graph/testgraph_7vertices.txt");

        IGraph<String, String> graph = GraphLoader
                        .loadUndirectedGraphEdgeListFile(cpr.getTempFileFromArchive().getAbsolutePath(), 7, ",");

        assertEquals(graph.numVertices(), 7);
        int[][] edges = {{1, 2}, {0, 2, 4}, {0, 1, 3, 4}, {2, 4, 5}, {1, 2, 3, 5, 6}, {3, 4, 6}, {4, 5}};

        for (int i = 0; i < 7; i++) {
            assertEquals(edges[i].length, graph.getVertexDegree(i));
            int[] connectedVertices = graph.getConnectedVertexIndices(i);
            for (int j = 0; j < edges[i].length; j++) {
                assertTrue(ArrayUtils.contains(connectedVertices, edges[i][j]));
            }
        }
    }

    @Test()
    @Timeout(10000)
    public void testGraphLoading() throws IOException {

        ClassPathResource cpr = new ClassPathResource("deeplearning4j-graph/simplegraph.txt");

        EdgeLineProcessor<String> edgeLineProcessor = new DelimitedEdgeLineProcessor(",", false, "//");
        VertexFactory<String> vertexFactory = new StringVertexFactory("v_%d");
        Graph<String, String> graph = GraphLoader.loadGraph(cpr.getTempFileFromArchive().getAbsolutePath(),
                        edgeLineProcessor, vertexFactory, 10, false);


//        System.out.println(graph);

        for (int i = 0; i < 10; i++) {
            List<Edge<String>> edges = graph.getEdgesOut(i);
            assertEquals(2, edges.size());

            //expect for example 0->1 and 9->0
            Edge<String> first = edges.get(0);
            if (first.getFrom() == i) {
                //undirected edge: i -> i+1 (or 9 -> 0)
                assertEquals(i, first.getFrom());
                assertEquals((i + 1) % 10, first.getTo());
            } else {
                //undirected edge: i-1 -> i (or 9 -> 0)
                assertEquals((i + 10 - 1) % 10, first.getFrom());
                assertEquals(i, first.getTo());
            }

            Edge<String> second = edges.get(1);
            assertNotEquals(first.getFrom(), second.getFrom());
            if (second.getFrom() == i) {
                //undirected edge: i -> i+1 (or 9 -> 0)
                assertEquals(i, second.getFrom());
                assertEquals((i + 1) % 10, second.getTo());
            } else {
                //undirected edge: i-1 -> i (or 9 -> 0)
                assertEquals((i + 10 - 1) % 10, second.getFrom());
                assertEquals(i, second.getTo());
            }
        }
    }

    @Test()
    @Timeout(10000)
    public void testGraphLoadingWithVertices() throws IOException {

        ClassPathResource verticesCPR = new ClassPathResource("deeplearning4j-graph/test_graph_vertices.txt");
        ClassPathResource edgesCPR = new ClassPathResource("deeplearning4j-graph/test_graph_edges.txt");


        EdgeLineProcessor<String> edgeLineProcessor = new DelimitedEdgeLineProcessor(",", false, "//");
        VertexLoader<String> vertexLoader = new DelimitedVertexLoader(":", "//");

        Graph<String, String> graph = GraphLoader.loadGraph(verticesCPR.getTempFileFromArchive().getAbsolutePath(),
                        edgesCPR.getTempFileFromArchive().getAbsolutePath(), vertexLoader, edgeLineProcessor, false);

//        System.out.println(graph);

        for (int i = 0; i < 10; i++) {
            List<Edge<String>> edges = graph.getEdgesOut(i);
            assertEquals(2, edges.size());

            //expect for example 0->1 and 9->0
            Edge<String> first = edges.get(0);
            if (first.getFrom() == i) {
                //undirected edge: i -> i+1 (or 9 -> 0)
                assertEquals(i, first.getFrom());
                assertEquals((i + 1) % 10, first.getTo());
            } else {
                //undirected edge: i-1 -> i (or 9 -> 0)
                assertEquals((i + 10 - 1) % 10, first.getFrom());
                assertEquals(i, first.getTo());
            }

            Edge<String> second = edges.get(1);
            assertNotEquals(first.getFrom(), second.getFrom());
            if (second.getFrom() == i) {
                //undirected edge: i -> i+1 (or 9 -> 0)
                assertEquals(i, second.getFrom());
                assertEquals((i + 1) % 10, second.getTo());
            } else {
                //undirected edge: i-1 -> i (or 9 -> 0)
                assertEquals((i + 10 - 1) % 10, second.getFrom());
                assertEquals(i, second.getTo());
            }
        }
    }


}
