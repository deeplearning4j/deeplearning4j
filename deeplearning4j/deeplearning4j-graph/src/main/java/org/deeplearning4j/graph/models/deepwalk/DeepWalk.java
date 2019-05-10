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

package org.deeplearning4j.graph.models.deepwalk;

import lombok.AllArgsConstructor;
import org.deeplearning4j.graph.api.IGraph;
import org.deeplearning4j.graph.api.IVertexSequence;
import org.deeplearning4j.graph.api.NoEdgeHandling;
import org.deeplearning4j.graph.iterator.GraphWalkIterator;
import org.deeplearning4j.graph.iterator.parallel.GraphWalkIteratorProvider;
import org.deeplearning4j.graph.iterator.parallel.RandomWalkGraphIteratorProvider;
import org.deeplearning4j.graph.models.embeddings.GraphVectorLookupTable;
import org.deeplearning4j.graph.models.embeddings.GraphVectorsImpl;
import org.deeplearning4j.graph.models.embeddings.InMemoryGraphLookupTable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.threadly.concurrent.PriorityScheduler;
import org.threadly.concurrent.future.FutureUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**Implementation of the DeepWalk graph vectorization model, based on the paper
 * <i>DeepWalk: Online Learning of Social Representations</i> by Perozzi, Al-Rfou & Skiena (2014),
 * <a href="http://arxiv.org/abs/1403.6652">http://arxiv.org/abs/1403.6652</a><br>
 * Similar to word2vec in nature, DeepWalk is an unsupervised learning algorithm that learns a vector representation
 * of each vertex in a graph. Vector representations are learned using walks (usually random walks) on the vertices in
 * the graph.<br>
 * Once learned, these vector representations can then be used for purposes such as classification, clustering, similarity
 * search, etc on the graph<br>
 * @author Alex Black
 */
public class DeepWalk<V, E> extends GraphVectorsImpl<V, E> {
    public static final int STATUS_UPDATE_FREQUENCY = 1000;
    private Logger log = LoggerFactory.getLogger(DeepWalk.class);

    private int vectorSize;
    private int windowSize;
    private double learningRate;
    private boolean initCalled = false;
    private long seed;
    private int nThreads = Runtime.getRuntime().availableProcessors();
    private transient AtomicLong walkCounter = new AtomicLong(0);

    public DeepWalk() {

    }

    public int getVectorSize() {
        return vectorSize;
    }

    public int getWindowSize() {
        return windowSize;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        if (lookupTable != null)
            lookupTable.setLearningRate(learningRate);
    }

    /** Initialize the DeepWalk model with a given graph. */
    public void initialize(IGraph<V, E> graph) {
        int nVertices = graph.numVertices();
        int[] degrees = new int[nVertices];
        for (int i = 0; i < nVertices; i++)
            degrees[i] = graph.getVertexDegree(i);
        initialize(degrees);
    }

    /** Initialize the DeepWalk model with a list of vertex degrees for a graph.<br>
     * Specifically, graphVertexDegrees[i] represents the vertex degree of the ith vertex<br>
     * vertex degrees are used to construct a binary (Huffman) tree, which is in turn used in
     * the hierarchical softmax implementation
     * @param graphVertexDegrees degrees of each vertex
     */
    public void initialize(int[] graphVertexDegrees) {
        log.info("Initializing: Creating Huffman tree and lookup table...");
        GraphHuffman gh = new GraphHuffman(graphVertexDegrees.length);
        gh.buildTree(graphVertexDegrees);
        lookupTable = new InMemoryGraphLookupTable(graphVertexDegrees.length, vectorSize, gh, learningRate);
        initCalled = true;
        log.info("Initialization complete");
    }

    /** Fit the model, in parallel.
     * This creates a set of GraphWalkIterators, which are then distributed one to each thread
     * @param graph Graph to fit
     * @param walkLength Length of rangom walks to generate
     */
    public void fit(IGraph<V, E> graph, int walkLength) {
        if (!initCalled)
            initialize(graph);
        //First: create iterators, one for each thread

        GraphWalkIteratorProvider<V> iteratorProvider = new RandomWalkGraphIteratorProvider<>(graph, walkLength, seed,
                        NoEdgeHandling.SELF_LOOP_ON_DISCONNECTED);

        fit(iteratorProvider);
    }

    /** Fit the model, in parallel, using a given GraphWalkIteratorProvider.<br>
     * This object is used to generate multiple GraphWalkIterators, which can then be distributed to each thread
     * to do in parallel<br>
     * Note that {@link #fit(IGraph, int)} will be more convenient in many cases<br>
     * Note that {@link #initialize(IGraph)} or {@link #initialize(int[])} <em>must</em> be called first.
     * @param iteratorProvider GraphWalkIteratorProvider
     * @see #fit(IGraph, int)
     */
    public void fit(GraphWalkIteratorProvider<V> iteratorProvider) {
        if (!initCalled)
            throw new UnsupportedOperationException("DeepWalk not initialized (call initialize before fit)");
        List<GraphWalkIterator<V>> iteratorList = iteratorProvider.getGraphWalkIterators(nThreads);

        PriorityScheduler scheduler = new PriorityScheduler(nThreads);

        List<Future<Void>> list = new ArrayList<>(iteratorList.size());
        //log.info("Fitting Graph with {} threads", Math.max(nThreads,iteratorList.size()));
        for (GraphWalkIterator<V> iter : iteratorList) {
            LearningCallable c = new LearningCallable(iter);
            list.add(scheduler.submit(c));
        }

        scheduler.shutdown();   // wont shutdown till complete

        try {
            FutureUtils.blockTillAllCompleteOrFirstError(list);
        } catch (InterruptedException e) {
            // should not be possible with blocking till scheduler terminates
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } catch (ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    /**Fit the DeepWalk model <b>using a single thread</b> using a given GraphWalkIterator. If parallel fitting is required,
     * {@link #fit(IGraph, int)} or {@link #fit(GraphWalkIteratorProvider)} should be used.<br>
     * Note that {@link #initialize(IGraph)} or {@link #initialize(int[])} <em>must</em> be called first.
     *
     * @param iterator iterator for graph walks
     */
    public void fit(GraphWalkIterator<V> iterator) {
        if (!initCalled)
            throw new UnsupportedOperationException("DeepWalk not initialized (call initialize before fit)");
        int walkLength = iterator.walkLength();

        while (iterator.hasNext()) {
            IVertexSequence<V> sequence = iterator.next();

            //Skipgram model:
            int[] walk = new int[walkLength + 1];
            int i = 0;
            while (sequence.hasNext())
                walk[i++] = sequence.next().vertexID();

            skipGram(walk);

            long iter = walkCounter.incrementAndGet();
            if (iter % STATUS_UPDATE_FREQUENCY == 0) {
                log.info("Processed {} random walks on graph", iter);
            }
        }
    }

    private void skipGram(int[] walk) {
        for (int mid = windowSize; mid < walk.length - windowSize; mid++) {
            for (int pos = mid - windowSize; pos <= mid + windowSize; pos++) {
                if (pos == mid)
                    continue;

                //pair of vertices: walk[mid] -> walk[pos]
                lookupTable.iterate(walk[mid], walk[pos]);
            }
        }
    }

    public GraphVectorLookupTable lookupTable() {
        return lookupTable;
    }


    public static class Builder<V, E> {
        private int vectorSize = 100;
        private long seed = System.currentTimeMillis();
        private double learningRate = 0.01;
        private int windowSize = 2;

        /** Sets the size of the vectors to be learned for each vertex in the graph */
        public Builder<V, E> vectorSize(int vectorSize) {
            this.vectorSize = vectorSize;
            return this;
        }

        /** Set the learning rate */
        public Builder<V, E> learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        /** Sets the window size used in skipgram model */
        public Builder<V, E> windowSize(int windowSize) {
            this.windowSize = windowSize;
            return this;
        }

        /** Seed for random number generation (used for repeatability).
         * Note however that parallel/async gradient descent might result in behaviour that
         * is not repeatable, in spite of setting seed
         */
        public Builder<V, E> seed(long seed) {
            this.seed = seed;
            return this;
        }

        public DeepWalk<V, E> build() {
            DeepWalk<V, E> dw = new DeepWalk<>();
            dw.vectorSize = vectorSize;
            dw.windowSize = windowSize;
            dw.learningRate = learningRate;
            dw.seed = seed;

            return dw;
        }
    }

    @AllArgsConstructor
    private class LearningCallable implements Callable<Void> {

        private final GraphWalkIterator<V> iterator;

        @Override
        public Void call() throws Exception {
            fit(iterator);

            return null;
        }
    }
}
