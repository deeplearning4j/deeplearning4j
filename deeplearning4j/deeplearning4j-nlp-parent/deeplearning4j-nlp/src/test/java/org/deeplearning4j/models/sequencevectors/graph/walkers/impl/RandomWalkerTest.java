/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.models.sequencevectors.graph.walkers.impl;

import org.deeplearning4j.models.sequencevectors.graph.enums.NoEdgeHandling;
import org.deeplearning4j.models.sequencevectors.graph.enums.WalkDirection;
import org.deeplearning4j.models.sequencevectors.graph.exception.NoEdgesException;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Graph;
import org.deeplearning4j.models.sequencevectors.graph.primitives.IGraph;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Vertex;
import org.deeplearning4j.models.sequencevectors.graph.vertex.AbstractVertexFactory;
import org.deeplearning4j.models.sequencevectors.graph.walkers.GraphWalker;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class RandomWalkerTest {

    private static IGraph<VocabWord, Double> graph;
    private static IGraph<VocabWord, Double> graphBig;
    private static IGraph<VocabWord, Double> graphDirected;

    protected static final Logger logger = LoggerFactory.getLogger(RandomWalkerTest.class);

    @Before
    public void setUp() throws Exception {
        if (graph == null) {
            graph = new Graph<>(10, false, new AbstractVertexFactory<VocabWord>());

            for (int i = 0; i < 10; i++) {
                graph.getVertex(i).setValue(new VocabWord(i, String.valueOf(i)));

                int x = i + 3;
                if (x >= 10)
                    x = 0;
                graph.addEdge(i, x, 1.0, false);
            }

            graphDirected = new Graph<>(10, false, new AbstractVertexFactory<VocabWord>());

            for (int i = 0; i < 10; i++) {
                graphDirected.getVertex(i).setValue(new VocabWord(i, String.valueOf(i)));

                int x = i + 3;
                if (x >= 10)
                    x = 0;
                graphDirected.addEdge(i, x, 1.0, true);
            }

            graphBig = new Graph<>(1000, false, new AbstractVertexFactory<VocabWord>());

            for (int i = 0; i < 1000; i++) {
                graphBig.getVertex(i).setValue(new VocabWord(i, String.valueOf(i)));

                int x = i + 3;
                if (x >= 1000)
                    x = 0;
                graphBig.addEdge(i, x, 1.0, false);
            }
        }
    }

    @Test
    public void testGraphCreation() throws Exception {
        Graph<VocabWord, Double> graph = new Graph<>(10, false, new AbstractVertexFactory<VocabWord>());

        // we have 10 elements
        assertEquals(10, graph.numVertices());

        for (int i = 0; i < 10; i++) {
            Vertex<VocabWord> vertex = graph.getVertex(i);
            assertEquals(null, vertex.getValue());
            assertEquals(i, vertex.vertexID());
        }
        assertEquals(10, graph.numVertices());
    }

    @Test
    public void testGraphTraverseRandom1() throws Exception {
        RandomWalker<VocabWord> walker = (RandomWalker<VocabWord>) new RandomWalker.Builder<>(graph)
                        .setNoEdgeHandling(NoEdgeHandling.SELF_LOOP_ON_DISCONNECTED).setWalkLength(3).build();

        int cnt = 0;
        while (walker.hasNext()) {
            Sequence<VocabWord> sequence = walker.next();

            assertEquals(3, sequence.getElements().size());
            assertNotEquals(null, sequence);

            for (VocabWord word : sequence.getElements()) {
                assertNotEquals(null, word);
            }

            cnt++;
        }

        assertEquals(10, cnt);
    }

    @Test
    public void testGraphTraverseRandom2() throws Exception {
        RandomWalker<VocabWord> walker = (RandomWalker<VocabWord>) new RandomWalker.Builder<>(graph)
                        .setNoEdgeHandling(NoEdgeHandling.EXCEPTION_ON_DISCONNECTED).setWalkLength(20)
                        .setWalkDirection(WalkDirection.FORWARD_UNIQUE)
                        .setNoEdgeHandling(NoEdgeHandling.CUTOFF_ON_DISCONNECTED).build();

        int cnt = 0;
        while (walker.hasNext()) {
            Sequence<VocabWord> sequence = walker.next();

            assertTrue(sequence.getElements().size() <= 10);
            assertNotEquals(null, sequence);

            for (VocabWord word : sequence.getElements()) {
                assertNotEquals(null, word);
            }

            cnt++;
        }

        assertEquals(10, cnt);
    }

    @Test
    public void testGraphTraverseRandom3() throws Exception {
        RandomWalker<VocabWord> walker = (RandomWalker<VocabWord>) new RandomWalker.Builder<>(graph)
                        .setNoEdgeHandling(NoEdgeHandling.EXCEPTION_ON_DISCONNECTED).setWalkLength(20)
                        .setWalkDirection(WalkDirection.FORWARD_UNIQUE)
                        .setNoEdgeHandling(NoEdgeHandling.EXCEPTION_ON_DISCONNECTED).build();

        try {
            while (walker.hasNext()) {
                Sequence<VocabWord> sequence = walker.next();
                logger.info("Sequence: " + sequence);
            }

            // if cycle passed without exception - something went bad
            assertTrue(false);
        } catch (NoEdgesException e) {
            // this cycle should throw exception
        } catch (Exception e) {
            assertTrue(false);
        }
    }

    @Test
    public void testGraphTraverseRandom4() throws Exception {
        RandomWalker<VocabWord> walker = (RandomWalker<VocabWord>) new RandomWalker.Builder<>(graphBig)
                        .setNoEdgeHandling(NoEdgeHandling.EXCEPTION_ON_DISCONNECTED).setWalkLength(20)
                        .setWalkDirection(WalkDirection.FORWARD_UNIQUE)
                        .setNoEdgeHandling(NoEdgeHandling.CUTOFF_ON_DISCONNECTED).build();

        Sequence<VocabWord> sequence1 = walker.next();

        walker.reset(true);

        Sequence<VocabWord> sequence2 = walker.next();

        assertNotEquals(sequence1.getElements().get(0), sequence2.getElements().get(0));
    }

    @Test
    public void testGraphTraverseRandom5() throws Exception {
        RandomWalker<VocabWord> walker = (RandomWalker<VocabWord>) new RandomWalker.Builder<>(graphBig)
                        .setWalkLength(20).setWalkDirection(WalkDirection.FORWARD_UNIQUE)
                        .setNoEdgeHandling(NoEdgeHandling.CUTOFF_ON_DISCONNECTED).build();

        Sequence<VocabWord> sequence1 = walker.next();

        walker.reset(false);

        Sequence<VocabWord> sequence2 = walker.next();

        assertEquals(sequence1.getElements().get(0), sequence2.getElements().get(0));
    }

    @Test
    public void testGraphTraverseRandom6() throws Exception {
        GraphWalker<VocabWord> walker = new RandomWalker.Builder<>(graphDirected).setWalkLength(20)
                        .setWalkDirection(WalkDirection.FORWARD_UNIQUE)
                        .setNoEdgeHandling(NoEdgeHandling.CUTOFF_ON_DISCONNECTED).build();

        Sequence<VocabWord> sequence = walker.next();
        assertEquals("0", sequence.getElements().get(0).getLabel());
        assertEquals("3", sequence.getElements().get(1).getLabel());
        assertEquals("6", sequence.getElements().get(2).getLabel());
        assertEquals("9", sequence.getElements().get(3).getLabel());

        assertEquals(4, sequence.getElements().size());
    }
}
