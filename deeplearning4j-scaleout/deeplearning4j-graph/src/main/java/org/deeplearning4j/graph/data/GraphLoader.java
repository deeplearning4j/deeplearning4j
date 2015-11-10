package org.deeplearning4j.graph.data;

import org.deeplearning4j.graph.api.Edge;
import org.deeplearning4j.graph.api.Vertex;
import org.deeplearning4j.graph.vertexfactory.VertexFactory;
import org.deeplearning4j.graph.graph.dl4j.SimpleGraph;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

public class GraphLoader {

    /** Load a graph into memory, using a EdgeLineProcessor.
     * Assume one edge per line
     * @param path Path to the file containing the edges, one per line
     * @param lineProcessor EdgeLineProcessor used to convert lines of text into a graph (or null for comment lines etc)
     * @param vertexFactory Used to create vertices
     * @return Graph
     */
    public static <V,E> SimpleGraph<V,E> loadGraph(String path, EdgeLineProcessor<E> lineProcessor,
                                                   VertexFactory<V> vertexFactory, int numVertices,
                                                   boolean allowMultipleEdges) throws IOException {
        SimpleGraph<V,E> graph = new SimpleGraph<>(numVertices,allowMultipleEdges,vertexFactory);

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
     * @return Graph loaded from files
     */
    public static <V,E> SimpleGraph<V,E> loadGraph(String vertexFilePath, String edgeFilePath, VertexLoader<V> vertexLoader,
                                                   EdgeLineProcessor<E> edgeLineProcessor, boolean allowMultipleEdges ) throws IOException {
        //Assume vertices are in one file
        //And edges are in another file

        List<Vertex<V>> vertices = vertexLoader.loadVertices(vertexFilePath);
        SimpleGraph<V,E> graph = new SimpleGraph<>(vertices,allowMultipleEdges);

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
