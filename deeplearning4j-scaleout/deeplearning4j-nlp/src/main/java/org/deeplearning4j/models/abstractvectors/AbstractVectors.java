package org.deeplearning4j.models.abstractvectors;

import lombok.NonNull;
import lombok.Setter;
import org.deeplearning4j.graph.iterator.GraphWalkIterator;
import org.deeplearning4j.models.abstractvectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.abstractvectors.sequence.SequenceElement;
import org.deeplearning4j.models.abstractvectors.transformers.SequenceTransformer;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * AbstractVectors implements abstract features extraction for Sequences and SequenceElements, using SkipGram, CBOW or DBOW (for Sequence features extraction).
 *
 * DO NOT USE, IT'S JUST A DRAFT FOR FUTURE WordVectorsImpl changes
 * @author raver119@gmail.com
 */
public class AbstractVectors<T extends SequenceElement> extends WordVectorsImpl<T> implements WordVectors {


    protected VectorsConfiguration configuration;

    /**
     * Starts training over
     */
    public void fit() {

    }

    public static class Builder<T extends SequenceElement> {
        protected double sampling;
        protected double negative;
        protected double learningRate;
        protected double minLearningRate;
        protected int minWordFrequency;
        protected VocabCache<T> vocabCache;
        protected WeightLookupTable<T> lookupTable;
        protected int iterations;
        protected int numEpochs;
        protected int layerSize;
        protected int window;
        protected boolean hugeModelExpected;
        protected int batchSize;
        protected int learningRateDecayWords;
        protected long seed;
        protected boolean useAdaGrad;
        protected boolean resetModel;

        protected List<String> stopWords = new ArrayList<>();

        protected VectorsConfiguration configuration;

        public Builder() {

        }

        public Builder(@NonNull VectorsConfiguration configuration) {
            this.configuration = configuration;
            this.iterations = configuration.getIterations();
            this.numEpochs = configuration.getEpochs();
            this.minLearningRate = configuration.getMinLearningRate();
            this.learningRate = configuration.getLearningRate();
            this.sampling = configuration.getSampling();
            this.negative = configuration.getNegative();
            this.minWordFrequency = configuration.getMinWordFrequency();
            this.seed = configuration.getSeed();
            this.hugeModelExpected = configuration.isHugeModelExpected();
            this.batchSize = configuration.getBatchSize();
            this.layerSize = configuration.getLayersSize();
            this.learningRateDecayWords = configuration.getLearningRateDecayWords();
            this.useAdaGrad = configuration.isUseAdaGrad();
            this.window = configuration.getWindow();
        }

        /**
         * Set sequence transformer for abstract transformations data
         *
         * @param transformer
         * @return
         */
        public Builder<T> setSequenceTransformer(@NonNull SequenceTransformer<T, ?> transformer) {

            return this;
        }

        public Builder<T> iterate(@NonNull SequenceIterator<T> iterator) {

            return this;
        }

        public Builder<T> setLayerSize(int layerSize) {
            this.layerSize = layerSize;
            return this;
        }

        public Builder<T> setLearningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder<T> minWordFrequency(int minWordFrequency) {
            this.minWordFrequency = minWordFrequency;
            return this;
        }

        public Builder<T> resetModel(boolean reallyReset) {
            this.resetModel = reallyReset;
            return this;
        }

        public Builder<T> setVocabCache(@NonNull VocabCache<T> vocabCache) {
            this.vocabCache = vocabCache;
            return this;
        }


        /**
         *  You can provide collection of objects to be ignored, and excluded out of model
         *  Please note: Object labels and hashCode will be used for filtering
         *
         * @param stopList
         * @return
         */
        public Builder<T> stopList(@NonNull List<String> stopList) {
            this.stopWords.addAll(stopList);
            return this;
        }

        /**
         * You can provide collection of objects to be ignored, and excluded out of model
         * Please note: Object labels and hashCode will be used for filtering
         *
         * @param stopList
         * @return
         */
        public Builder<T> stopList(@NonNull Collection<T> stopList) {
            for (T word: stopList) {
                this.stopWords.add(word.getLabel());
            }
            return this;
        }

        public AbstractVectors<T> build() {
            AbstractVectors<T> vectors = new AbstractVectors<>();
            vectors.numEpochs = this.numEpochs;
            vectors.numIterations = this.iterations;
            vectors.vocab = this.vocabCache;
            vectors.minWordFrequency = this.minWordFrequency;
            vectors.learningRate.set(this.learningRate);
            vectors.minLearningRate = this.minLearningRate;
            vectors.sampling = this.sampling;
            vectors.negative = this.negative;
            vectors.layerSize = this.layerSize;
            vectors.batchSize = this.batchSize;
            vectors.learningRateDecayWords = this.learningRateDecayWords;
            vectors.window = this.window;
            vectors.resetModel = this.resetModel;
            vectors.useAdeGrad = this.useAdaGrad;
            vectors.stopWords = this.stopWords;


            this.configuration.setLearningRate(this.learningRate);
            this.configuration.setLayersSize(layerSize);
            this.configuration.setHugeModelExpected(hugeModelExpected);
            this.configuration.setWindow(window);
            this.configuration.setMinWordFrequency(minWordFrequency);
            this.configuration.setIterations(iterations);
            this.configuration.setSeed(seed);
            this.configuration.setBatchSize(batchSize);
            this.configuration.setLearningRateDecayWords(learningRateDecayWords);
            this.configuration.setMinLearningRate(minLearningRate);
            this.configuration.setSampling(this.sampling);
            this.configuration.setUseAdaGrad(useAdaGrad);
            this.configuration.setNegative(negative);
            this.configuration.setEpochs(this.numEpochs);

            vectors.configuration = this.configuration;

            return vectors;
        }
    }
}
