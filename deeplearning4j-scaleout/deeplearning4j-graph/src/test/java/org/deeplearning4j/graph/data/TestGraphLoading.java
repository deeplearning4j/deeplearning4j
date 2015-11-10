package org.deeplearning4j.graph.data;

import org.deeplearning4j.graph.api.Edge;
import org.deeplearning4j.graph.data.impl.DelimitedEdgeLineProcessor;
import org.deeplearning4j.graph.data.impl.DelimitedVertexLoader;
import org.deeplearning4j.graph.graph.dl4j.SimpleGraph;
import org.deeplearning4j.graph.vertexfactory.StringVertexFactory;
import org.deeplearning4j.graph.vertexfactory.VertexFactory;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class TestGraphLoading {

    @Test
    public void testGraphLoading() throws IOException{

        ClassPathResource cpr = new ClassPathResource("simplegraph.txt");

        EdgeLineProcessor<String> edgeLineProcessor = new DelimitedEdgeLineProcessor(",",false,"//");
        VertexFactory<String> vertexFactory = new StringVertexFactory("v_%d");
        SimpleGraph<String,String> graph = GraphLoader.loadGraph(cpr.getFile().getAbsolutePath(),
                edgeLineProcessor,vertexFactory,10,false);


        System.out.println(graph);

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
            assertNotEquals(first.getFrom(),second.getFrom());
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

    @Test
    public void testGraphLoadingWithVertices() throws IOException {

        ClassPathResource verticesCPR = new ClassPathResource("test_graph_vertices.txt");
        ClassPathResource edgesCPR = new ClassPathResource("test_graph_edges.txt");


        EdgeLineProcessor<String> edgeLineProcessor = new DelimitedEdgeLineProcessor(",",false,"//");
        VertexLoader<String> vertexLoader = new DelimitedVertexLoader(":","//");

        SimpleGraph<String,String> graph = GraphLoader.loadGraph(verticesCPR.getFile().getAbsolutePath(),
                edgesCPR.getFile().getAbsolutePath(),vertexLoader,edgeLineProcessor,false);

        System.out.println(graph);

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
            assertNotEquals(first.getFrom(),second.getFrom());
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
