package org.deeplearning4j.models.word2vec;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.StreamLineIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.util.Collection;
import java.util.List;

/**
 * This is Word2Vec implementation based on SequenceVectors
 *
 * @author raver119@gmail.com
 */
public class Word2Vec extends SequenceVectors<VocabWord> {
    @Getter protected SentenceIterator sentenceIter;
    @Getter @Setter protected TokenizerFactory tokenizerFactory;

    public void setSentenceIter(@NonNull SentenceIterator iterator) {
        if (tokenizerFactory == null) throw new IllegalStateException("Please call setTokenizerFactory() prior to setSentenceIter() call.");

        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(iterator)
                .tokenizerFactory(tokenizerFactory)
                .build();
        this.iterator = new AbstractSequenceIterator.Builder<VocabWord>(transformer).build();
    }

    public static class Builder extends SequenceVectors.Builder<VocabWord> {
        protected SentenceIterator sentenceIterator;
        protected TokenizerFactory tokenizerFactory;


        public Builder() {

        }

        public Builder(@NonNull VectorsConfiguration configuration) {
            super(configuration);
        }

        public Builder iterate(@NonNull DocumentIterator iterator) {
            this.sentenceIterator = new StreamLineIterator.Builder(iterator)
                    .setFetchSize(100)
                    .build();
            return this;
        }


        public Builder iterate(@NonNull SentenceIterator iterator) {
            this.sentenceIterator = iterator;
            return this;
        }

        public Builder tokenizerFactory(@NonNull TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        @Deprecated
        public Builder index(@NonNull InvertedIndex<VocabWord> index) {
            return this;
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
            super.lookupTable(lookupTable);
            return this;
        }

        @Override
        public Builder sampling(double sampling) {
            super.sampling(sampling);
            return this;
        }

        @Override
        public Builder useAdaGrad(boolean reallyUse) {
            super.useAdaGrad(reallyUse);
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
            throw new IllegalStateException("You can't change this option for Word2Vec");
        }

        @Override
        public Builder trainSequencesRepresentation(boolean trainSequences) {
            throw new IllegalStateException("You can't change this option for Word2Vec");
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

        public Word2Vec build() {
            Word2Vec ret = new Word2Vec();

            if (tokenizerFactory == null) tokenizerFactory = new DefaultTokenizerFactory();

            if (sentenceIterator != null) {
                SentenceTransformer transformer = new SentenceTransformer.Builder()
                        .iterator(sentenceIterator)
                        .tokenizerFactory(tokenizerFactory)
                        .build();
                this.iterator = new AbstractSequenceIterator.Builder<VocabWord>(transformer).build();
            }

            ret.numEpochs = this.numEpochs;
            ret.numIterations = this.iterations;
            ret.vocab = this.vocabCache;
            ret.minWordFrequency = this.minWordFrequency;
            ret.learningRate.set(this.learningRate);
            ret.minLearningRate = this.minLearningRate;
            ret.sampling = this.sampling;
            ret.negative = this.negative;
            ret.layerSize = this.layerSize;
            ret.batchSize = this.batchSize;
            ret.learningRateDecayWords = this.learningRateDecayWords;
            ret.window = this.window;
            ret.resetModel = this.resetModel;
            ret.useAdeGrad = this.useAdaGrad;
            ret.stopWords = this.stopWords;
            ret.workers = this.workers;

            ret.iterator = this.iterator;
            ret.lookupTable = this.lookupTable;
            ret.tokenizerFactory = this.tokenizerFactory;

            ret.elementsLearningAlgorithm = this.elementsLearningAlgorithm;
            ret.sequenceLearningAlgorithm = this.sequenceLearningAlgorithm;

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

            ret.configuration = this.configuration;

            // we hardcode
            ret.trainSequenceVectors = false;
            ret.trainElementsVectors = true;

            return ret;
        }
    }
}
