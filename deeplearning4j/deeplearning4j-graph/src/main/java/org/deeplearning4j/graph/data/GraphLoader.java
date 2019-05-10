/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.graph.data;

import org.deeplearning4j.graph.api.Edge;
import org.deeplearning4j.graph.api.Vertex;
import org.deeplearning4j.graph.data.impl.DelimitedEdgeLineProcessor;
import org.deeplearning4j.graph.data.impl.WeightedEdgeLineProcessor;
import org.deeplearning4j.graph.graph.Graph;
import org.deeplearning4j.graph.vertexfactory.StringVertexFactory;
import org.deeplearning4j.graph.vertexfactory.VertexFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

/** Utility methods for loading graphs
 *
 */
public class GraphLoader {

    private GraphLoader() {}

    /** Simple method for loading an undirected graph, where the graph is represented by a edge list with one edge
     * per line with a delimiter in between<br>
     * This method assumes that all lines in the file are of the form {@code i<delim>j} where i and j are integers
     * in range 0 to numVertices inclusive, and "<delim>" is the user-provided delimiter
     * <b>Note</b>: this method calls {@link #loadUndirectedGraphEdgeListFile(String, int, String, boolean)} with allowMultipleEdges = true.
     * @param path Path to the edge list file
     * @param numVertices number of vertices in the graph
     * @return graph
     * @throws IOException if file cannot be read
     */
    public static Graph<String, String> loadUndirectedGraphEdgeListFile(String path, int numVertices, String delim)
                    throws IOException {
        return loadUndirectedGraphEdgeListFile(path, numVertices, delim, true);
    }

    /** Simple method for loading an undirected graph, where the graph is represented by a edge list with one edge
     * per line with a delimiter in between<br>
     * This method assumes that all lines in the file are of the form {@code i<delim>j} where i and j are integers
     * in range 0 to numVertices inclusive, and "<delim>" is the user-provided delimiter
     * @param path Path to the edge list file
     * @param numVertices number of vertices in the graph
     * @param allowMultipleEdges If set to false, the graph will not allow multiple edges between any two vertices to exist. However,
     *                           checking for duplicates during graph loading can be costly, so use allowMultipleEdges=true when
     *                           possible.
     * @return graph
     * @throws IOException if file cannot be read
     */
    public static Graph<String, String> loadUndirectedGraphEdgeListFile(String path, int numVertices, String delim,
                    boolean allowMultipleEdges) throws IOException {
        Graph<String, String> graph = new Graph<>(numVertices, allowMultipleEdges, new StringVertexFactory());
        EdgeLineProcessor<String> lineProcessor = new DelimitedEdgeLineProcessor(delim, false);

        try (BufferedReader br = new BufferedReader(new FileReader(new File(path)))) {
            String line;
            while ((line = br.readLine()) != null) {
                Edge<String> edge = lineProcessor.processLine(line);
                if (edge != null) {
                    graph.addEdge(edge);
                }
            }
        }
        return graph;
    }

    /**Method for loading a weighted graph from an edge list file, where each edge (inc. weight) is represented by a
     * single line. Graph may be directed or undirected<br>
     * This method assumes that edges are of the format: {@code fromIndex<delim>toIndex<delim>edgeWeight} where {@code <delim>}
     * is the delimiter.
     * <b>Note</b>: this method calls {@link #loadWeightedEdgeListFile(String, int, String, boolean, boolean, String...)} with allowMultipleEdges = true.
     * @param path Path to the edge list file
     * @param numVertices The number of vertices in the graph
     * @param delim The delimiter used in the file (typically: "," or " " etc)
     * @param directed whether the edges should be treated as directed (true) or undirected (false)
     * @param ignoreLinesStartingWith Starting characters for comment lines. May be null. For example: "//" or "#"
     * @return The graph
     * @throws IOException
     */
    public static Graph<String, Double> loadWeightedEdgeListFile(String path, int numVertices, String delim,
                    boolean directed, String... ignoreLinesStartingWith) throws IOException {
        return loadWeightedEdgeListFile(path, numVertices, delim, directed, true, ignoreLinesStartingWith);
    }

    /**Method for loading a weighted graph from an edge list file, where each edge (inc. weight) is represented by a
     * single line. Graph may be directed or undirected<br>
     * This method assumes that edges are of the format: {@code fromIndex<delim>toIndex<delim>edgeWeight} where {@code <delim>}
     * is the delimiter.
     * @param path Path to the edge list file
     * @param numVertices The number of vertices in the graph
     * @param delim The delimiter used in the file (typically: "," or " " etc)
     * @param directed whether the edges should be treated as directed (true) or undirected (false)
     * @param allowMultipleEdges If set to false, the graph will not allow multiple edges between any two vertices to exist. However,
     *                           checking for duplicates during graph loading can be costly, so use allowMultipleEdges=true when
     *                           possible.
     * @param ignoreLinesStartingWith Starting characters for comment lines. May be null. For example: "//" or "#"
     * @return The graph
     * @throws IOException
     */
    public static Graph<String, Double> loadWeightedEdgeListFile(String path, int numVertices, String delim,
                    boolean directed, boolean allowMultipleEdges, String... ignoreLinesStartingWith)
                    throws IOException {
        Graph<String, Double> graph = new Graph<>(numVertices, allowMultipleEdges, new StringVertexFactory());
        EdgeLineProcessor<Double> lineProcessor =
                        new WeightedEdgeLineProcessor(delim, directed, ignoreLinesStartingWith);

        try (BufferedReader br = new BufferedReader(new FileReader(new File(path)))) {
            String line;
            while ((line = br.readLine()) != null) {
                Edge<Double> edge = lineProcessor.processLine(line);
                if (edge != null) {
                    graph.addEdge(edge);
                }
            }
        }
        return graph;
    }

    /** Load a graph into memory, using a given EdgeLineProcessor.
     * Assume one edge per line
     * @param path Path to the file containing the edges, one per line
     * @param lineProcessor EdgeLineProcessor used to convert lines of text into a graph (or null for comment lines etc)
     * @param vertexFactory Used to create vertices
     * @param numVertices number of vertices in the graph
     * @param allowMultipleEdges whether the graph should allow multiple edges between a given pair of vertices or not
     * @return IGraph
     */
    public static <V, E> Graph<V, E> loadGraph(String path, EdgeLineProcessor<E> lineProcessor,
                    VertexFactory<V> vertexFactory, int numVertices, boolean allowMultipleEdges) throws IOException {
        Graph<V, E> graph = new Graph<>(numVertices, allowMultipleEdges, vertexFactory);

        try (BufferedReader br = new BufferedReader(new FileReader(new File(path)))) {
            String line;
            while ((line = br.readLine()) != null) {
                Edge<E> edge = lineProcessor.processLine(line);
                if (edge != null) {
                    graph.addEdge(edge);
                }
            }
        }

        return graph;
    }

    /** Load graph, assuming vertices are in one file and edges are in another file.
     *
     * @param vertexFilePath Path to file containing vertices, one per line
     * @param edgeFilePath Path to the file containing edges, one per line
     * @param vertexLoader VertexLoader, for loading vertices from the file
     * @param edgeLineProcessor EdgeLineProcessor, converts text lines into edges
     * @param allowMultipleEdges whether the graph should allow (or filter out) multiple edges
     * @return IGraph loaded from files
     */
    public static <V, E> Graph<V, E> loadGraph(String vertexFilePath, String edgeFilePath, VertexLoader<V> vertexLoader,
                    EdgeLineProcessor<E> edgeLineProcessor, boolean allowMultipleEdges) throws IOException {
        //Assume vertices are in one file
        //And edges are in another file

        List<Vertex<V>> vertices = vertexLoader.loadVertices(vertexFilePath);
        Graph<V, E> graph = new Graph<>(vertices, allowMultipleEdges);

        try (BufferedReader br = new BufferedReader(new FileReader(new File(edgeFilePath)))) {
            String line;
            while ((line = br.readLine()) != null) {
                Edge<E> edge = edgeLineProcessor.processLine(line);
                if (edge != null) {
                    graph.addEdge(edge);
                }
            }
        }

        return graph;
    }
}
