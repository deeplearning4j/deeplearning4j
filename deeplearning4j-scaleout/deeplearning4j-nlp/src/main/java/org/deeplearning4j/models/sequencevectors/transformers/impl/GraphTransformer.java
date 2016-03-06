package org.deeplearning4j.models.sequencevectors.transformers.impl;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.graph.enums.NoEdgeHandling;
import org.deeplearning4j.models.sequencevectors.graph.enums.WalkDirection;
import org.deeplearning4j.models.sequencevectors.graph.enums.WalkMode;
import org.deeplearning4j.models.sequencevectors.graph.huffman.GraphHuffman;
import org.deeplearning4j.models.sequencevectors.graph.primitives.IGraph;
import org.deeplearning4j.models.sequencevectors.graph.walkers.GraphWalker;
import org.deeplearning4j.models.sequencevectors.graph.walkers.RandomWalker;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.text.labels.LabelsProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 *
 * This class is used to build vocabulary and sequences out of graph, via GraphWalkers
 *
 * WORK IS IN PROGRESS, DO NOT USE
 * @author raver119@gmail.com
 */
public class GraphTransformer<T extends SequenceElement> implements Iterable<Sequence<T>> {
    protected IGraph<T, ?> sourceGraph;
    protected int walkLength = 5;
    protected WalkMode walkMode = WalkMode.RANDOM;
    protected NoEdgeHandling noEdgeHandling = NoEdgeHandling.EXCEPTION_ON_DISCONNECTED;
    protected GraphWalker<T> walker;
    protected WalkDirection walkDirection = WalkDirection.FORWARD_ONLY;
    protected LabelsProvider<T> labelsProvider;
    protected AtomicInteger counter = new AtomicInteger(0);
    protected boolean shuffle = true;

    protected static final Logger log = LoggerFactory.getLogger(GraphTransformer.class);

    protected GraphTransformer() {
        ;
    }

    /**
     * This method handles required initialization for GraphTransformer
     */
    protected void initialize() {
        log.info("Building Huffman tree for source graph...");
        int nVertices = sourceGraph.numVertices();
        int[] degrees = new int[nVertices];
        for( int i=0; i<nVertices; i++ )
            degrees[i] = sourceGraph.getVertexDegree(i);

        GraphHuffman huffman = new GraphHuffman(nVertices);
        huffman.buildTree(degrees);

        log.info("Transferring Huffman tree info to nodes...");
        for (int i = 0; i < nVertices; i++) {
            int codeLen = huffman.getCodeLength(i);
            int[] path = huffman.getPathInnerNodes(i);
            List<Integer> codes = huffman.getCodeList(i);

            T element = sourceGraph.getVertex(i).getValue();
            element.setCodeLength(codeLen);
            element.setPoints(path);
            element.setCodes(codes);
        }
    }

    @Override
    public Iterator<Sequence<T>> iterator() {
        this.counter.set(0);
        this.walker.reset();
        return new Iterator<Sequence<T>>() {
            private GraphWalker<T> walker = GraphTransformer.this.walker;

            @Override
            public boolean hasNext() {
                return walker.hasNext();
            }

            @Override
            public Sequence<T> next() {
                Sequence<T> sequence = walker.next();
                sequence.setSequenceId(counter.getAndIncrement());

                if (labelsProvider != null) {
                    // TODO: sequence labels to be implemented for graph walks
                    sequence.setSequenceLabel(labelsProvider.getLabel(sequence.getSequenceId()));
                }
                return sequence;
            }
        };
    }

    public static class Builder<T extends SequenceElement> {
        protected IGraph<T, ?> sourceGraph;
        protected int walkLength = 5;
        protected WalkMode walkMode = WalkMode.RANDOM;
        protected NoEdgeHandling noEdgeHandling = NoEdgeHandling.CUTOFF_ON_DISCONNECTED;
        protected WalkDirection walkDirection = WalkDirection.FORWARD_ONLY;
        protected LabelsProvider<T> labelsProvider;
        protected GraphWalker<T> walker;
        protected boolean shuffle = true;

        public Builder(IGraph<T, ?> sourceGraph) {
            this.sourceGraph = sourceGraph;
        }

        public Builder<T> setNoEdgeHandling(@NonNull NoEdgeHandling handling) {
            this.noEdgeHandling = handling;
            return this;
        }

        public Builder<T> setWalkMode(@NonNull WalkMode walkMode) {
            this.walkMode = walkMode;
            return this;
        }

        public Builder<T> setWalkLength(int walkLength) {
            this.walkLength = walkLength;
            return this;
        }

        public Builder<T> setLabelsProvider(@NonNull LabelsProvider<T> provider) {
            this.labelsProvider = provider;
            return this;
        }

        public Builder<T> setGraphWalker(@NonNull GraphWalker<T> walker) {
            this.walker = walker;
            return this;
        }

        public Builder<T> shuffleOnReset(boolean reallyShuffle) {
            this.shuffle = reallyShuffle;
            return this;
        }

        public GraphTransformer<T> build() {
            GraphTransformer<T> transformer = new GraphTransformer<T>();
            transformer.noEdgeHandling = this.noEdgeHandling;
            transformer.sourceGraph = this.sourceGraph;
            transformer.walkLength = this.walkLength;
            transformer.walkMode = this.walkMode;
            transformer.walkDirection = this.walkDirection;
            transformer.labelsProvider = this.labelsProvider;
            transformer.shuffle = this.shuffle;

            if (this.walker == null)
                switch (this.walkMode) {
                    case RANDOM:
                        transformer.walker = new RandomWalker.Builder<T>(this.sourceGraph)
                                .setNoEdgeHandling(this.noEdgeHandling)
                                .setWalkLength(this.walkLength)
                                .setWalkDirection(this.walkDirection)
                                .build();
                        break;
                    case WEIGHTED_MAX:
                    case WEIGHTED_MIN:
                    default:
                        throw new UnsupportedOperationException("WalkMode ["+ this.walkMode+"] isn't supported at this moment?");
                }
            else transformer.walker = this.walker;

            transformer.initialize();

            return transformer;
        }
    }
}
