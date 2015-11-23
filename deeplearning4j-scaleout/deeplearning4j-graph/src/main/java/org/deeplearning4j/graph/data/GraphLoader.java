package org.deeplearning4j.graph.data;

import org.deeplearning4j.graph.api.Edge;
import org.deeplearning4j.graph.api.Vertex;
import org.deeplearning4j.graph.data.impl.DelimitedEdgeLineProcessor;
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

    /** Simple method for loading an undirected graph, where the graph is represented by a edge list with one edge
     * per line with a delimiter in between<br>
     * This method assumes that all lines in the file are of the form "i<delim>j" where i and j are integers
     * in range 0 to numVertices inclusive, and "<delim>" is the user-provided delimiter
     * @param path Path to the edge list file
     * @param numVertices number of vertices in the graph
     * @return graph
     * @throws IOException if file cannot be read
     */
    public static Graph<String,String> loadUndirectedGraphEdgeListFile(String path, int numVertices, String delim) throws IOException{
        Graph<String,String> graph = new Graph<>(numVertices,false,new StringVertexFactory());
        EdgeLineProcessor<String> lineProcessor = new DelimitedEdgeLineProcessor(delim,false);

        try(BufferedReader br = new BufferedReader(new FileReader(new File(path)))){
            String line;
            while( (line = br.readLine()) != null ) {
                Edge<String> edge = lineProcessor.processLine(line);
                if(edge != null){
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
    public static <V,E> Graph<V,E> loadGraph(String path, EdgeLineProcessor<E> lineProcessor,
                                                   VertexFactory<V> vertexFactory, int numVertices,
                                                   boolean allowMultipleEdges) throws IOException {
        Graph<V,E> graph = new Graph<>(numVertices,allowMultipleEdges,vertexFactory);

        try(BufferedReader br = new BufferedReader(new FileReader(new File(path)))){
            String line;
            while( (line = br.readLine()) != null ) {
                Edge<E> edge = lineProcessor.processLine(line);
                if(edge != null){
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
    public static <V,E> Graph<V,E> loadGraph(String vertexFilePath, String edgeFilePath, VertexLoader<V> vertexLoader,
                                                   EdgeLineProcessor<E> edgeLineProcessor, boolean allowMultipleEdges ) throws IOException {
        //Assume vertices are in one file
        //And edges are in another file

        List<Vertex<V>> vertices = vertexLoader.loadVertices(vertexFilePath);
        Graph<V,E> graph = new Graph<>(vertices,allowMultipleEdges);

        try(BufferedReader br = new BufferedReader(new FileReader(new File(edgeFilePath)))){
            String line;
            while( (line = br.readLine()) != null ) {
                Edge<E> edge = edgeLineProcessor.processLine(line);
                if(edge != null){
                    graph.addEdge(edge);
                }
            }
        }

        return graph;
    }
}
