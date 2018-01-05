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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * @author raver119@gmail.com
 */
public class WeightedWalkerTest {
    private static Graph<VocabWord, Integer> basicGraph;

    @Before
    public void setUp() throws Exception {
        if (basicGraph == null) {
            // we don't really care about this graph, since it's just basic graph for iteration checks
            basicGraph = new Graph<>(10, false, new AbstractVertexFactory<VocabWord>());

            for (int i = 0; i < 10; i++) {
                basicGraph.getVertex(i).setValue(new VocabWord(i, String.valueOf(i)));

                int x = i + 3;
                if (x >= 10)
                    x = 0;
                basicGraph.addEdge(i, x, 1, false);
            }

            basicGraph.addEdge(0, 4, 2, false);
            basicGraph.addEdge(0, 4, 4, false);
            basicGraph.addEdge(0, 4, 6, false);
            basicGraph.addEdge(4, 5, 8, false);
            basicGraph.addEdge(1, 3, 6, false);
            basicGraph.addEdge(9, 7, 4, false);
            basicGraph.addEdge(5, 6, 2, false);
        }
    }

    @Test
    public void testBasicIterator1() throws Exception {
        GraphWalker<VocabWord> walker = new WeightedWalker.Builder<>(basicGraph)
                        .setWalkDirection(WalkDirection.FORWARD_PREFERRED).setWalkLength(10)
                        .setNoEdgeHandling(NoEdgeHandling.RESTART_ON_DISCONNECTED).build();

        int cnt = 0;
        while (walker.hasNext()) {
            Sequence<VocabWord> sequence = walker.next();

            assertNotEquals(null, sequence);
            assertEquals(10, sequence.getElements().size());
            cnt++;
        }

        assertEquals(basicGraph.numVertices(), cnt);
    }

}
