package org.deeplearning4j.models.glove;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import java.util.Collection;
import java.util.List;

/**
 *
 * WORK IN PROGRESS, PLEASE DO NOT USE
 * @author raver119@gmail.com
 */
public class Glove extends SequenceVectors<VocabWord> {

    protected Glove() {

    }

    public static class Builder extends SequenceVectors.Builder<VocabWord> {
        public Builder() {
            super();
        }

        public Builder(@NonNull VectorsConfiguration configuration) {
            super(configuration);
        }

        @Override
        public Builder iterate(@NonNull SequenceIterator<VocabWord> iterator) {
            super.iterate(iterator);
            return this;
        }

        @Override
        public Builder batchSize(int batchSize) {
            super.batchSize(batchSize);
            return this;
        }

        @Override
        public Builder iterations(int iterations) {
            super.iterations(iterations);
            return this;
        }

        @Override
        public Builder epochs(int numEpochs) {
            super.epochs(numEpochs);
            return this;
        }

        @Override
        public Builder useAdaGrad(boolean reallyUse) {
            super.useAdaGrad(reallyUse);
            return this;
        }

        @Override
        public Builder layerSize(int layerSize) {
            super.layerSize(layerSize);
            return this;
        }

        @Override
        public Builder learningRate(double learningRate) {
            super.learningRate(learningRate);
            return this;
        }

        @Override
        public Builder minWordFrequency(int minWordFrequency) {
            super.minWordFrequency(minWordFrequency);
            return this;
        }

        @Override
        public Builder minLearningRate(double minLearningRate) {
            super.minLearningRate(minLearningRate);
            return this;
        }

        @Override
        public Builder resetModel(boolean reallyReset) {
            super.resetModel(reallyReset);
            return this;
        }

        @Override
        public Builder vocabCache(@NonNull VocabCache<VocabWord> vocabCache) {
            super.vocabCache(vocabCache);
            return this;
        }

        @Override
        public Builder lookupTable(@NonNull WeightLookupTable<VocabWord> lookupTable) {
            if (! (lookupTable instanceof GloveWeightLookupTable)) throw new IllegalStateException("You need to use GloveWeightLookupTable for Glove implementation");

            super.lookupTable(lookupTable);
            return this;
        }

        @Override
        public Builder sampling(double sampling) {
            super.sampling(sampling);
            return this;
        }

        @Override
        public Builder negativeSample(double negative) {
            super.negativeSample(negative);
            return this;
        }

        @Override
        public Builder stopWords(@NonNull List<String> stopList) {
            super.stopWords(stopList);
            return this;
        }

        @Override
        public Builder trainElementsRepresentation(boolean trainElements) {
            super.trainElementsRepresentation(trainElements);
            return this;
        }

        @Override
        @Deprecated
        public Builder trainSequencesRepresentation(boolean trainSequences) {
            super.trainSequencesRepresentation(trainSequences);
            return this;
        }

        @Override
        public Builder stopWords(@NonNull Collection<VocabWord> stopList) {
            super.stopWords(stopList);
            return this;
        }

        @Override
        public Builder windowSize(int windowSize) {
            super.windowSize(windowSize);
            return this;
        }

        @Override
        public Builder seed(long randomSeed) {
            super.seed(randomSeed);
            return this;
        }

        @Override
        public Builder workers(int numWorkers) {
            super.workers(numWorkers);
            return this;
        }

        public Glove build() {
            Glove ret = new Glove();

            // hardcoded value for glove


            return ret;
        }
    }
}
