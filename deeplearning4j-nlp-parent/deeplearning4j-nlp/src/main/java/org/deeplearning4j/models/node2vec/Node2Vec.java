package org.deeplearning4j.models.node2vec;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.learning.SequenceLearningAlgorithm;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Vertex;
import org.deeplearning4j.models.sequencevectors.graph.walkers.GraphWalker;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.interfaces.VectorsListener;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.sequencevectors.transformers.impl.GraphTransformer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.List;

/**
 * This is implementation for Node2Vec/DeepWalk for DeepLearning4J
 *
 * PLEASE NOTE: This class is under construction and isn't suited for any use.
 *
 * @author raver119@gmail.com
 */
@Slf4j
@Deprecated
public class Node2Vec<V extends SequenceElement, E extends Number> extends SequenceVectors<V> {

    public INDArray inferVector(@NonNull Collection<Vertex<V>> vertices) {
        return null;
    }

    public static class Builder<V extends SequenceElement, E extends Number> extends SequenceVectors.Builder<V> {
        private GraphWalker<V> walker;

        public Builder(@NonNull GraphWalker<V> walker, @NonNull VectorsConfiguration configuration) {
            this.walker = walker;
            this.configuration = configuration;

            // FIXME: this will cause transformer initialization
            GraphTransformer<V> transformer = new GraphTransformer.Builder<>(walker.getSourceGraph())
                            .setGraphWalker(walker).shuffleOnReset(true).build();

            this.iterator = new AbstractSequenceIterator.Builder<V>(transformer).build();
        }


        @Override
        protected Builder<V, E> useExistingWordVectors(@NonNull WordVectors vec) {
            super.useExistingWordVectors(vec);
            return this;
        }

        @Override
        public Builder<V, E> iterate(@NonNull SequenceIterator<V> iterator) {
            super.iterate(iterator);
            return this;
        }

        @Override
        public Builder<V, E> sequenceLearningAlgorithm(@NonNull String algoName) {
            super.sequenceLearningAlgorithm(algoName);
            return this;
        }

        @Override
        public Builder<V, E> sequenceLearningAlgorithm(@NonNull SequenceLearningAlgorithm<V> algorithm) {
            super.sequenceLearningAlgorithm(algorithm);
            return this;
        }

        @Override
        public Builder<V, E> elementsLearningAlgorithm(@NonNull String algoName) {
            super.elementsLearningAlgorithm(algoName);
            return this;
        }

        @Override
        public Builder<V, E> elementsLearningAlgorithm(@NonNull ElementsLearningAlgorithm<V> algorithm) {
            super.elementsLearningAlgorithm(algorithm);
            return this;
        }

        @Override
        public Builder<V, E> iterations(int iterations) {
            super.iterations(iterations);
            return this;
        }

        @Override
        public Builder<V, E> epochs(int numEpochs) {
            super.epochs(numEpochs);
            return this;
        }

        @Override
        public Builder<V, E> workers(int numWorkers) {
            super.workers(numWorkers);
            return this;
        }

        @Override
        public Builder<V, E> useHierarchicSoftmax(boolean reallyUse) {
            super.useHierarchicSoftmax(reallyUse);
            return this;
        }

        @Override
        public Builder<V, E> useAdaGrad(boolean reallyUse) {
            super.useAdaGrad(reallyUse);
            return this;
        }

        @Override
        public Builder<V, E> layerSize(int layerSize) {
            super.layerSize(layerSize);
            return this;
        }

        @Override
        public Builder<V, E> learningRate(double learningRate) {
            super.learningRate(learningRate);
            return this;
        }

        @Override
        public Builder<V, E> minWordFrequency(int minWordFrequency) {
            super.minWordFrequency(minWordFrequency);
            return this;
        }

        @Override
        public Builder<V, E> minLearningRate(double minLearningRate) {
            super.minLearningRate(minLearningRate);
            return this;
        }

        @Override
        public Builder<V, E> resetModel(boolean reallyReset) {
            super.resetModel(reallyReset);
            return this;
        }

        @Override
        public Builder<V, E> vocabCache(@NonNull VocabCache<V> vocabCache) {
            super.vocabCache(vocabCache);
            return this;
        }

        @Override
        public Builder<V, E> lookupTable(@NonNull WeightLookupTable<V> lookupTable) {
            super.lookupTable(lookupTable);
            return this;
        }

        @Override
        public Builder<V, E> sampling(double sampling) {
            super.sampling(sampling);
            return this;
        }

        @Override
        public Builder<V, E> negativeSample(double negative) {
            super.negativeSample(negative);
            return this;
        }

        @Override
        public Builder<V, E> stopWords(@NonNull List<String> stopList) {
            super.stopWords(stopList);
            return this;
        }

        @Override
        public Builder<V, E> trainElementsRepresentation(boolean trainElements) {
            super.trainElementsRepresentation(trainElements);
            return this;
        }

        @Override
        public Builder<V, E> trainSequencesRepresentation(boolean trainSequences) {
            super.trainSequencesRepresentation(trainSequences);
            return this;
        }

        @Override
        public Builder<V, E> stopWords(@NonNull Collection<V> stopList) {
            super.stopWords(stopList);
            return this;
        }

        @Override
        public Builder<V, E> windowSize(int windowSize) {
            super.windowSize(windowSize);
            return this;
        }

        @Override
        public Builder<V, E> seed(long randomSeed) {
            super.seed(randomSeed);
            return this;
        }

        @Override
        public Builder<V, E> modelUtils(@NonNull ModelUtils<V> modelUtils) {
            super.modelUtils(modelUtils);
            return this;
        }

        @Override
        public Builder<V, E> useUnknown(boolean reallyUse) {
            super.useUnknown(reallyUse);
            return this;
        }

        @Override
        public Builder<V, E> unknownElement(@NonNull V element) {
            super.unknownElement(element);
            return this;
        }

        @Override
        public Builder<V, E> useVariableWindow(int... windows) {
            super.useVariableWindow(windows);
            return this;
        }

        @Override
        public Builder<V, E> usePreciseWeightInit(boolean reallyUse) {
            super.usePreciseWeightInit(reallyUse);
            return this;
        }

        @Override
        protected void presetTables() {
            super.presetTables();
        }

        @Override
        public Builder<V, E> setVectorsListeners(@NonNull Collection<VectorsListener<V>> vectorsListeners) {
            super.setVectorsListeners(vectorsListeners);
            return this;
        }

        @Override
        public Builder<V, E> enableScavenger(boolean reallyEnable) {
            super.enableScavenger(reallyEnable);
            return this;
        }

        public Node2Vec<V, E> build() {
            Node2Vec<V, E> node2vec = new Node2Vec<>();
            node2vec.iterator = this.iterator;

            return node2vec;
        }
    }
}
