package org.deeplearning4j.models.sequencevectors.graph.walkers.impl;

import org.deeplearning4j.models.sequencevectors.graph.enums.NoEdgeHandling;
import org.deeplearning4j.models.sequencevectors.graph.enums.PopularityMode;
import org.deeplearning4j.models.sequencevectors.graph.enums.SpreadSpectrum;
import org.deeplearning4j.models.sequencevectors.graph.enums.WalkDirection;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Graph;
import org.deeplearning4j.models.sequencevectors.graph.vertex.AbstractVertexFactory;
import org.deeplearning4j.models.sequencevectors.graph.walkers.GraphWalker;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.junit.Before;
import org.junit.Test;

import java.util.concurrent.atomic.AtomicBoolean;

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
            graph.addEdge(1, 3, 1.0, false);
            graph.addEdge(9, 7, 1.0, false);
            graph.addEdge(5, 6, 1.0, false);
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
                .setNoEdgeHandling(NoEdgeHandling.CUTOFF_ON_DISCONNECTED)
                .setWalkLength(10)
                .setPopularityMode(PopularityMode.MAXIMUM)
                .setPopularitySpread(3)
                .setSpreadSpectrum(SpreadSpectrum.PLAIN)
                .build();

        System.out.println("Connected [3] size: " + graph.getConnectedVertices(3).size());
        System.out.println("Connected [4] size: " + graph.getConnectedVertices(4).size());

        Sequence<VocabWord> sequence = walker.next();
        assertEquals("0", sequence.getElements().get(0).getLabel());

        System.out.println("Position at 1: [" + sequence.getElements().get(1).getLabel() + "]");

        assertTrue(sequence.getElements().get(1).getLabel().equals("4") || sequence.getElements().get(1).getLabel().equals("7") || sequence.getElements().get(1).getLabel().equals("9"));
    }

    @Test
    public void testPopularityWalker2() throws Exception {
        GraphWalker<VocabWord> walker = new PopularityWalker.Builder<VocabWord>(graph)
                .setWalkDirection(WalkDirection.FORWARD_ONLY)
                .setNoEdgeHandling(NoEdgeHandling.CUTOFF_ON_DISCONNECTED)
                .setWalkLength(10)
                .setPopularityMode(PopularityMode.MINIMUM)
                .setPopularitySpread(3)
                .build();

        System.out.println("Connected [3] size: " + graph.getConnectedVertices(3).size());
        System.out.println("Connected [4] size: " + graph.getConnectedVertices(4).size());

        Sequence<VocabWord> sequence = walker.next();
        assertEquals("0", sequence.getElements().get(0).getLabel());

        System.out.println("Position at 1: [" + sequence.getElements().get(1).getLabel() + "]");

        assertTrue(sequence.getElements().get(1).getLabel().equals("8") || sequence.getElements().get(1).getLabel().equals("3") || sequence.getElements().get(1).getLabel().equals("9")  || sequence.getElements().get(1).getLabel().equals("7"));
    }

    @Test
    public void testPopularityWalker3() throws Exception {
        GraphWalker<VocabWord> walker = new PopularityWalker.Builder<VocabWord>(graph)
                .setWalkDirection(WalkDirection.FORWARD_ONLY)
                .setNoEdgeHandling(NoEdgeHandling.CUTOFF_ON_DISCONNECTED)
                .setWalkLength(10)
                .setPopularityMode(PopularityMode.MAXIMUM)
                .setPopularitySpread(3)
                .setSpreadSpectrum(SpreadSpectrum.PROPORTIONAL)
                .build();

        System.out.println("Connected [3] size: " + graph.getConnectedVertices(3).size());
        System.out.println("Connected [4] size: " + graph.getConnectedVertices(4).size());

        AtomicBoolean got4 = new AtomicBoolean(false);
        AtomicBoolean got7 = new AtomicBoolean(false);
        AtomicBoolean got9 = new AtomicBoolean(false);

        for (int i = 0; i < 50; i++) {
            Sequence<VocabWord> sequence = walker.next();
            assertEquals("0", sequence.getElements().get(0).getLabel());
            System.out.println("Position at 1: [" + sequence.getElements().get(1).getLabel() + "]");

            got4.compareAndSet(false, sequence.getElements().get(1).getLabel().equals("4"));
            got7.compareAndSet(false, sequence.getElements().get(1).getLabel().equals("7"));
            got9.compareAndSet(false, sequence.getElements().get(1).getLabel().equals("9"));

            assertTrue(sequence.getElements().get(1).getLabel().equals("4") || sequence.getElements().get(1).getLabel().equals("7") || sequence.getElements().get(1).getLabel().equals("9"));

            walker.reset(false);
        }

        assertTrue(got4.get());
        assertTrue(got7.get());
        assertTrue(got9.get());
    }

    @Test
    public void testPopularityWalker4() throws Exception {
        GraphWalker<VocabWord> walker = new PopularityWalker.Builder<VocabWord>(graph)
                .setWalkDirection(WalkDirection.FORWARD_ONLY)
                .setNoEdgeHandling(NoEdgeHandling.CUTOFF_ON_DISCONNECTED)
                .setWalkLength(10)
                .setPopularityMode(PopularityMode.MINIMUM)
                .setPopularitySpread(3)
                .setSpreadSpectrum(SpreadSpectrum.PROPORTIONAL)
                .build();

        System.out.println("Connected [3] size: " + graph.getConnectedVertices(3).size());
        System.out.println("Connected [4] size: " + graph.getConnectedVertices(4).size());

        AtomicBoolean got3 = new AtomicBoolean(false);
        AtomicBoolean got8 = new AtomicBoolean(false);
        AtomicBoolean got9 = new AtomicBoolean(false);

        for (int i = 0; i < 50; i++) {
            Sequence<VocabWord> sequence = walker.next();
            assertEquals("0", sequence.getElements().get(0).getLabel());
            System.out.println("Position at 1: [" + sequence.getElements().get(1).getLabel() + "]");

            got3.compareAndSet(false, sequence.getElements().get(1).getLabel().equals("3"));
            got8.compareAndSet(false, sequence.getElements().get(1).getLabel().equals("8"));
            got9.compareAndSet(false, sequence.getElements().get(1).getLabel().equals("9"));

            assertTrue(sequence.getElements().get(1).getLabel().equals("8") || sequence.getElements().get(1).getLabel().equals("3") || sequence.getElements().get(1).getLabel().equals("9"));

            walker.reset(false);
        }

        assertTrue(got3.get());
        assertTrue(got8.get());
        assertTrue(got9.get());
    }
}