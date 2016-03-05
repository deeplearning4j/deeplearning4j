package org.deeplearning4j.models.sequencevectors.transformers.impl;

import org.deeplearning4j.models.sequencevectors.graph.enums.NoEdgeHandling;
import org.deeplearning4j.models.sequencevectors.graph.enums.WalkDirection;
import org.deeplearning4j.models.sequencevectors.graph.exception.NoEdgesException;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Graph;
import org.deeplearning4j.models.sequencevectors.graph.primitives.IGraph;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Vertex;
import org.deeplearning4j.models.sequencevectors.graph.vertex.AbstractVertexFactory;
import org.deeplearning4j.models.sequencevectors.graph.walkers.RandomWalker;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class GraphTransformerTest {
    private static IGraph<VocabWord, Void> graph;

    @Before
    public void setUp() throws Exception {
        if (graph == null) {
            graph = new Graph<VocabWord, Void>(10, false, new AbstractVertexFactory<VocabWord>());

            for (int i = 0; i < 10; i++) {
                graph.getVertex(i).setValue(new VocabWord(i, String.valueOf(i)));

                int x = i + 3;
                if (x >= 10) x = 0;
                graph.addEdge(i, x, null, true);
            }
        }
    }

    @Test
    public void testGraphCreation() throws Exception {
        Graph<VocabWord, Void> graph = new Graph<VocabWord, Void>(10, false, new AbstractVertexFactory<VocabWord>());

        // we have 10 elements
        assertEquals(10,graph.numVertices());

        for (int i = 0; i < 10; i++) {
            Vertex<VocabWord> vertex = graph.getVertex(i);
            assertEquals(null, vertex.getValue());
            assertEquals(i, vertex.vertexID());
        }
        assertEquals(10, graph.numVertices());
    }

    @Test
    public void testGraphTraverseRandom1() throws Exception {
        RandomWalker<VocabWord> walker = new RandomWalker.Builder<VocabWord>(graph)
                .setNoEdgeHandling(NoEdgeHandling.EXCEPTION_ON_DISCONNECTED)
                .setWalkLength(3)
                .build();

        int cnt = 0;
        while (walker.hasNext()) {
            Sequence<VocabWord> sequence = walker.next();

            assertEquals(3, sequence.getElements().size());
            assertNotEquals(null, sequence);

            for (VocabWord word: sequence.getElements()) {
                assertNotEquals(null, word);
            }

            cnt++;
        }

        assertEquals(10, cnt);
    }

    @Test
    public void testGraphTraverseRandom2() throws Exception {
        RandomWalker<VocabWord> walker = new RandomWalker.Builder<VocabWord>(graph)
                .setNoEdgeHandling(NoEdgeHandling.EXCEPTION_ON_DISCONNECTED)
                .setWalkLength(20)
                .setWalkDirection(WalkDirection.FORWARD_UNIQUE)
                .setNoEdgeHandling(NoEdgeHandling.CUTOFF_ON_DISCONNECTED)
                .build();

        int cnt = 0;
        while (walker.hasNext()) {
            Sequence<VocabWord> sequence = walker.next();

            assertTrue(sequence.getElements().size() <= 10);
            assertNotEquals(null, sequence);

            for (VocabWord word: sequence.getElements()) {
                assertNotEquals(null, word);
            }

            cnt++;
        }

        assertEquals(10, cnt);
    }

    @Test
    public void testGraphTraverseRandom3() throws Exception {
        RandomWalker<VocabWord> walker = new RandomWalker.Builder<VocabWord>(graph)
                .setNoEdgeHandling(NoEdgeHandling.EXCEPTION_ON_DISCONNECTED)
                .setWalkLength(20)
                .setWalkDirection(WalkDirection.FORWARD_UNIQUE)
                .setNoEdgeHandling(NoEdgeHandling.EXCEPTION_ON_DISCONNECTED)
                .build();

        try {
            while (walker.hasNext()) {
                Sequence<VocabWord> sequence = walker.next();
            }

            // if cycle passed without exception - something went bad
            assertTrue(false);
        } catch (NoEdgesException e) {
            // this cycle should throw exception
        } catch (Exception e) {
            assertTrue(false);
        }
    }
}