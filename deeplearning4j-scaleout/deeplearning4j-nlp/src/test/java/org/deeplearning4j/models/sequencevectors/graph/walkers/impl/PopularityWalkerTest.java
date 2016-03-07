package org.deeplearning4j.models.sequencevectors.graph.walkers.impl;

import org.deeplearning4j.models.sequencevectors.graph.enums.NoEdgeHandling;
import org.deeplearning4j.models.sequencevectors.graph.enums.WalkDirection;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Graph;
import org.deeplearning4j.models.sequencevectors.graph.vertex.AbstractVertexFactory;
import org.deeplearning4j.models.sequencevectors.graph.walkers.GraphWalker;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class PopularityWalkerTest {

    private static Graph<VocabWord, Double> graph;

    @Before
    public void setUp() {
        if (graph == null) {
            graph = new Graph<VocabWord, Double>(10, false, new AbstractVertexFactory<VocabWord>());

            for (int i = 0; i < 10; i++) {
                graph.getVertex(i).setValue(new VocabWord(i, String.valueOf(i)));

                int x = i + 3;
                if (x >= 10) x = 0;
                graph.addEdge(i, x, 1.0, false);
            }

            graph.addEdge(0, 4, 1.0, false);
            graph.addEdge(0, 4, 1.0, false);
            graph.addEdge(0, 4, 1.0, false);
            graph.addEdge(4, 5, 1.0, false);
        }
    }

    @Test
    public void testPopularityWalkerCreation() throws Exception {
        GraphWalker<VocabWord> walker = new PopularityWalker.Builder<VocabWord>(graph)
                .setWalkDirection(WalkDirection.FORWARD_ONLY)
                .setWalkLength(10)
                .setNoEdgeHandling(NoEdgeHandling.CUTOFF_ON_DISCONNECTED)
                .build();

        assertEquals("PopularityWalker", walker.getClass().getSimpleName());
    }

    @Test
    public void testPopularityWalker1() throws Exception {
        GraphWalker<VocabWord> walker = new PopularityWalker.Builder<VocabWord>(graph)
                .setWalkDirection(WalkDirection.FORWARD_ONLY)
                .setWalkLength(10)
                .build();

        System.out.println("Connected [3] size: " + graph.getConnectedVertices(3).size());
        System.out.println("Connected [4] size: " + graph.getConnectedVertices(4).size());

        Sequence<VocabWord> sequence = walker.next();
        assertEquals("0", sequence.getElements().get(0).getLabel());
        assertEquals("4", sequence.getElements().get(1).getLabel());
    }
}