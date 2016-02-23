package org.deeplearning4j.models.word2vec;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
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
    @Getter protected transient SentenceIterator sentenceIter;
    @Getter protected transient TokenizerFactory tokenizerFactory;

    /**
     * This method defines TokenizerFactory instance to be using during model building
     *
     * @param tokenizerFactory TokenizerFactory instance
     */
    public void setTokenizerFactory(@NonNull TokenizerFactory tokenizerFactory) {
        this.tokenizerFactory = tokenizerFactory;

        if (sentenceIter != null) {
            SentenceTransformer transformer = new SentenceTransformer.Builder()
                    .iterator(sentenceIter)
                    .tokenizerFactory(this.tokenizerFactory)
                    .build();
            this.iterator = new AbstractSequenceIterator.Builder<VocabWord>(transformer).build();
        }
    }

    /**
     * This method defines SentenceIterator instance, that will be used as training corpus source
     *
     * @param iterator SentenceIterator instance
     */
    public void setSentenceIter(@NonNull SentenceIterator iterator) {
        //if (tokenizerFactory == null) throw new IllegalStateException("Please call setTokenizerFactory() prior to setSentenceIter() call.");

        if (tokenizerFactory != null) {
            SentenceTransformer transformer = new SentenceTransformer.Builder()
                    .iterator(iterator)
                    .tokenizerFactory(tokenizerFactory)
                    .build();
            this.iterator = new AbstractSequenceIterator.Builder<VocabWord>(transformer).build();
        }
    }

    public static class Builder extends SequenceVectors.Builder<VocabWord> {
        protected SentenceIterator sentenceIterator;
        protected TokenizerFactory tokenizerFactory;


        public Builder() {

        }

        /**
         * This method has no effect for Word2Vec
         *
         * @param vec existing WordVectors model
         * @return
         */
        @Override
        protected Builder useExistingWordVectors(@NonNull WordVectors vec) {
            return this;
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

        /**
         * This method used to feed SentenceIterator, that contains training corpus, into ParagraphVectors
         *
         * @param iterator
         * @return
         */
        public Builder iterate(@NonNull SentenceIterator iterator) {
            this.sentenceIterator = iterator;
            return this;
        }

        /**
         * This method defines TokenizerFactory to be used for strings tokenization during training
         * PLEASE NOTE: If external VocabCache is used, the same TokenizerFactory should be used to keep derived tokens equal.
         *
         * @param tokenizerFactory
         * @return
         */
        public Builder tokenizerFactory(@NonNull TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        @Deprecated
        public Builder index(@NonNull InvertedIndex<VocabWord> index) {
            return this;
        }

        /**
         * This method used to feed SequenceIterator, that contains training corpus, into ParagraphVectors
         *
         * @param iterator
         * @return
         */
        @Override
        public Builder iterate(@NonNull SequenceIterator<VocabWord> iterator) {
            super.iterate(iterator);
            return this;
        }

        /**
         * This method defines mini-batch size
         * @param batchSize
         * @return
         */
        @Override
        public Builder batchSize(int batchSize) {
            super.batchSize(batchSize);
            return this;
        }

        /**
         * This method defines number of iterations done for each mini-batch during training
         * @param iterations
         * @return
         */
        @Override
        public Builder iterations(int iterations) {
            super.iterations(iterations);
            return this;
        }

        /**
         * This method defines number of epochs (iterations over whole training corpus) for training
         * @param numEpochs
         * @return
         */
        @Override
        public Builder epochs(int numEpochs) {
            super.epochs(numEpochs);
            return this;
        }

        /**
         * This method defines number of dimensions for output vectors
         * @param layerSize
         * @return
         */
        @Override
        public Builder layerSize(int layerSize) {
            super.layerSize(layerSize);
            return this;
        }

        /**
         * This method defines initial learning rate for model training
         *
         * @param learningRate
         * @return
         */
        @Override
        public Builder learningRate(double learningRate) {
            super.learningRate(learningRate);
            return this;
        }

        /**
         * This method defines minimal word frequency in training corpus. All words below this threshold will be removed prior model training
         *
         * @param minWordFrequency
         * @return
         */
        @Override
        public Builder minWordFrequency(int minWordFrequency) {
            super.minWordFrequency(minWordFrequency);
            return this;
        }

        /**
         * This method defines minimal learning rate value for training
         *
         * @param minLearningRate
         * @return
         */
        @Override
        public Builder minLearningRate(double minLearningRate) {
            super.minLearningRate(minLearningRate);
            return this;
        }

        /**
         * This method defines whether model should be totally wiped out prior building, or not
         *
         * @param reallyReset
         * @return
         */
        @Override
        public Builder resetModel(boolean reallyReset) {
            super.resetModel(reallyReset);
            return this;
        }

        /**
         * This method allows to define external VocabCache to be used
         *
         * @param vocabCache
         * @return
         */
        @Override
        public Builder vocabCache(@NonNull VocabCache<VocabWord> vocabCache) {
            super.vocabCache(vocabCache);
            return this;
        }

        /**
         * This method allows to define external WeightLookupTable to be used
         *
         * @param lookupTable
         * @return
         */
        @Override
        public Builder lookupTable(@NonNull WeightLookupTable<VocabWord> lookupTable) {
            super.lookupTable(lookupTable);
            return this;
        }

        /**
         * This method defines whether subsampling should be used or not
         *
         * @param sampling set > 0 to subsampling argument, or 0 to disable
         * @return
         */
        @Override
        public Builder sampling(double sampling) {
            super.sampling(sampling);
            return this;
        }

        /**
         * This method defines whether adaptive gradients should be used or not
         *
         * @param reallyUse
         * @return
         */
        @Override
        public Builder useAdaGrad(boolean reallyUse) {
            super.useAdaGrad(reallyUse);
            return this;
        }

        /**
         * This method defines whether negative sampling should be used or not
         *
         * @param negative set > 0 as negative sampling argument, or 0 to disable
         * @return
         */
        @Override
        public Builder negativeSample(double negative) {
            super.negativeSample(negative);
            return this;
        }

        /**
         * This method defines stop words that should be ignored during training
         *
         * @param stopList
         * @return
         */
        @Override
        public Builder stopWords(@NonNull List<String> stopList) {
            super.stopWords(stopList);
            return this;
        }

        /**
         * This method is hardcoded to TRUE, since that's whole point of Word2Vec
         *
         * @param trainElements
         * @return
         */
        @Override
        public Builder trainElementsRepresentation(boolean trainElements) {
            throw new IllegalStateException("You can't change this option for Word2Vec");
        }

        /**
         * This method is hardcoded to FALSE, since that's whole point of Word2Vec
         *
         * @param trainSequences
         * @return
         */
        @Override
        public Builder trainSequencesRepresentation(boolean trainSequences) {
            throw new IllegalStateException("You can't change this option for Word2Vec");
        }

        /**
         * This method defines stop words that should be ignored during training
         *
         * @param stopList
         * @return
         */
        @Override
        public Builder stopWords(@NonNull Collection<VocabWord> stopList) {
            super.stopWords(stopList);
            return this;
        }

        /**
         * This method defines context window size
         *
         * @param windowSize
         * @return
         */
        @Override
        public Builder windowSize(int windowSize) {
            super.windowSize(windowSize);
            return this;
        }

        /**
         * This method defines random seed for random numbers generator
         * @param randomSeed
         * @return
         */
        @Override
        public Builder seed(long randomSeed) {
            super.seed(randomSeed);
            return this;
        }

        /**
         * This method defines maximum number of concurrent threads available for training
         *
         * @param numWorkers
         * @return
         */
        @Override
        public Builder workers(int numWorkers) {
            super.workers(numWorkers);
            return this;
        }

        /**
         * Sets ModelUtils that gonna be used as provider for utility methods: similarity(), wordsNearest(), accuracy(), etc
         *
         * @param modelUtils model utils to be used
         * @return
         */
        @Override
        public Builder modelUtils(@NonNull ModelUtils<VocabWord> modelUtils) {
            super.modelUtils(modelUtils);
            return this;
        }

        public Word2Vec build() {
            presetTables();

            Word2Vec ret = new Word2Vec();

            if (sentenceIterator != null) {
                if (tokenizerFactory == null) tokenizerFactory = new DefaultTokenizerFactory();

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
            ret.modelUtils = this.modelUtils;

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
            this.configuration.setStopList(this.stopWords);

            ret.configuration = this.configuration;

            // we hardcode
            ret.trainSequenceVectors = false;
            ret.trainElementsVectors = true;

            return ret;
        }
    }
}
