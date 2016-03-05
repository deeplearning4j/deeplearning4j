package org.deeplearning4j.models.sequencevectors.transformers.impl;

import org.deeplearning4j.models.sequencevectors.graph.primitives.Graph;
import org.deeplearning4j.models.sequencevectors.graph.primitives.IGraph;
import org.deeplearning4j.models.sequencevectors.graph.vertex.AbstractVertexFactory;
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
            graph = new Graph<VocabWord, Void>();
        }
    }

    @Test
    public void testGraphCreation() throws Exception {
        Graph<VocabWord, Void> graph = new Graph<VocabWord, Void>(10, false, new AbstractVertexFactory<VocabWord>());

        // we have 10 elements
        assertEquals(10,graph.numVertices());

        for (int i = 0; i < 10; i++) {
            //VocabWord word = new VocabWord(1, String.valueOf(i));

        }
        assertEquals(10, graph.numVertices());
    }
}