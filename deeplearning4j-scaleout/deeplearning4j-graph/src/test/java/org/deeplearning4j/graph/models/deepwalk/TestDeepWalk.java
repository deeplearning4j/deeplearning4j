package org.deeplearning4j.graph.models.deepwalk;

import org.deeplearning4j.graph.data.GraphLoader;
import org.deeplearning4j.graph.graph.Graph;
import org.deeplearning4j.graph.graph.RandomWalkIterator;
import org.deeplearning4j.graph.iterator.GraphWalkIterator;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;

public class TestDeepWalk {

    @Test
    public void testBasic() throws IOException{
        //Very basic test. Load graph, build tree, call fit, make sure it doesn't throw any exceptions

        ClassPathResource cpr = new ClassPathResource("testgraph_7vertices.txt");

        Graph<String,String> graph = GraphLoader.loadUndirectedGraphEdgeListFile(cpr.getFile().getAbsolutePath(), 7, ",");

        int vectorSize = 5;
        int windowSize = 2;

        DeepWalk<String,String> deepWalk = new DeepWalk.Builder<String,String>().learningRate(0.01)
                .vectorSize(vectorSize)
                .windowSize(windowSize)
                .learningRate(0.01)
                .build();
        deepWalk.initialize(graph);

        for( int i=0; i<7; i++ ){
            INDArray vector = deepWalk.getVertexVector(i);
            assertArrayEquals(new int[]{1,vectorSize},vector.shape());
            System.out.println(Arrays.toString(vector.dup().data().asFloat()));
        }

        GraphWalkIterator<String> iter = new RandomWalkIterator<>(graph,8);

        deepWalk.fit(iter);

        for( int t=0; t<5; t++ ) {
            iter.reset();
            deepWalk.fit(iter);
            System.out.println("--------------------");
            for (int i = 0; i < 7; i++) {
                INDArray vector = deepWalk.getVertexVector(i);
                assertArrayEquals(new int[]{1, vectorSize}, vector.shape());
                System.out.println(Arrays.toString(vector.dup().data().asFloat()));
            }
        }

    }

}
