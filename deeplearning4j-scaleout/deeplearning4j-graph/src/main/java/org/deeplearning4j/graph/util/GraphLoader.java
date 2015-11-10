package org.deeplearning4j.graph.util;

import org.deeplearning4j.graph.api.Edge;
import org.deeplearning4j.graph.api.Graph;
import org.deeplearning4j.graph.api.VertexFactory;
import org.deeplearning4j.graph.graph.dl4j.SimpleGraph;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

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

        //Q: how to handle the fact that we might need BOTH vertex and edge files?
        //i.e., one file contains vertex info, another contains edge info
        //Two methods
        //Method 1: (this method): provide a vertex factory + number of vertices
        //Method 2: provide a vertex processor to parse vertex lines

        //Q: how to handle mapping from whatever to integer?
        //For example, might have data in format "0,1" or "'User1' 'User2'" etc
        //Option 1: assume data is ALWAYS preprocessed in format "0 <delim> 1 ..." format
        //Option 2: make it the responsibility of EdgeLineProcessor to always output correct indices?
        //Option 3:

        //Q: how to handle number of vertices?
        //Option 1: User provides it
        //Option 2: Implement a way to infer this from the file

        Graph<V,E> graph = new SimpleGraph<>(numVertices,allowMultipleEdges,vertexFactory);

        try(BufferedReader br = new BufferedReader(new FileReader(new File(path)))){
            String line;
            while( (line = br.readLine()) != null ) {
                Edge<E> edge = lineProcessor.processLine(line);
                if(edge != null){
                    graph.addEdge(edge);
                }
            }
        }


        throw new UnsupportedOperationException();
    }


    public static <V,E> SimpleGraph<V,E> loadGraph(String vertexFilePath, String edgeFilePath, VertexLoader<V> vertexLoader,
                                                   EdgeLineProcessor<E> edgeLineProcessor, boolean allowMultipleEdges ){

        //Assume vertices are in one file
        //And edges are in another file

        throw new UnsupportedOperationException("Not yet implemented");
    }

}
