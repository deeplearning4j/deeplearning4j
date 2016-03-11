package org.deeplearning4j.models.glove;

import lombok.NonNull;
import org.deeplearning4j.models.embeddings.learning.impl.elements.GloVe;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.StreamLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.util.Collection;
import java.util.List;

/**
 * GlobalVectors standalone implementation for DL4j.
 * Based on original Stanford GloVe
 * http://www-nlp.stanford.edu/pubs/glove.pdf
 *
 * @author raver119@gmail.com
 */
public class Glove extends SequenceVectors<VocabWord> {

    protected Glove() {

    }

    public static class Builder extends SequenceVectors.Builder<VocabWord> {
        private double xMax;
        private boolean shuffle;
        private boolean symmetric;
        protected double alpha = 0.75d;
        private int maxmemory = (int) (Runtime.getRuntime().totalMemory() / 1024 /1024 / 1024);

        protected TokenizerFactory tokenFactory;
        protected SentenceIterator sentenceIterator;
        protected DocumentIterator documentIterator;

        public Builder() {
            super();
        }


        public Builder(@NonNull VectorsConfiguration configuration) {
            super(configuration);
        }


        /**
         * This method has no effect for GloVe
         *
         * @param vec existing WordVectors model
         * @return
         */
        @Override
        public Builder useExistingWordVectors(@NonNull WordVectors vec) {
            return this;
        }

        @Override
        public Builder iterate(@NonNull SequenceIterator<VocabWord> iterator) {
            super.iterate(iterator);
            return this;
        }

        /**
         * Specifies minibatch size for training process.
         *
         * @param batchSize
         * @return
         */
        @Override
        public Builder batchSize(int batchSize) {
            super.batchSize(batchSize);
            return this;
        }

        /**
         * Ierations and epochs are the same in GloVe implementation.
         *
         * @param iterations
         * @return
         */
        @Override
        public Builder iterations(int iterations) {
            super.epochs(iterations);
            return this;
        }

        /**
         * Sets the number of iteration over training corpus during training
         *
         * @param numEpochs
         * @return
         */
        @Override
        public Builder epochs(int numEpochs) {
            super.epochs(numEpochs);
            return this;
        }

        @Override
        public Builder useAdaGrad(boolean reallyUse) {
            super.useAdaGrad(true);
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

        /**
         * Sets minimum word frequency during vocabulary mastering.
         * Please note: this option is ignored, if vocabulary is built outside of GloVe
         *
         * @param minWordFrequency
         * @return
         */
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
        @Deprecated
        public Builder sampling(double sampling) {
            super.sampling(sampling);
            return this;
        }

        @Override
        @Deprecated
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
            super.trainElementsRepresentation(true);
            return this;
        }

        @Override
        @Deprecated
        public Builder trainSequencesRepresentation(boolean trainSequences) {
            super.trainSequencesRepresentation(false);
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

        /**
         * Sets TokenizerFactory to be used for training
         *
         * @param tokenizerFactory
         * @return
         */
        public Builder tokenizerFactory(@NonNull TokenizerFactory tokenizerFactory) {
            this.tokenFactory = tokenizerFactory;
            return this;
        }

        /**
         * Parameter specifying cutoff in weighting function; default 100.0
         *
         * @param xMax
         * @return
         */
        public Builder xMax(double xMax) {
            this.xMax = xMax;
            return this;
        }

        /**
         * Parameters specifying, if cooccurrences list should be build into both directions from any current word.
         *
         * @param reallySymmetric
         * @return
         */
        public Builder symmetric(boolean reallySymmetric) {
            this.symmetric = reallySymmetric;
            return this;
        }

        /**
         * Parameter specifying, if cooccurrences list should be shuffled between training epochs
         *
         * @param reallyShuffle
         * @return
         */
        public Builder shuffle(boolean reallyShuffle) {
            this.shuffle = reallyShuffle;
            return this;
        }

        /**
         * Parameter in exponent of weighting function; default 0.75
         *
         * @param alpha
         * @return
         */
        public Builder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }

        public Builder iterate(@NonNull SentenceIterator iterator) {
            this.sentenceIterator = iterator;
            return this;
        }

        public Builder iterate(@NonNull DocumentIterator iterator) {
            this.sentenceIterator = new StreamLineIterator.Builder(iterator)
                    .setFetchSize(100)
                    .build();
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

        /**
         * This method allows you to specify maximum memory available for CoOccurrence map builder.
         *
         * Please note: this option can be considered a debugging method. In most cases setting proper -Xmx argument set to JVM is enough to limit this algorithm.
         * Please note: this option won't override -Xmx JVM value.
         *
         * @param gbytes memory limit, in gigabytes
         * @return
         */
        public Builder maxMemory(int gbytes) {
            this.maxmemory = gbytes;
            return this;
        }

        /**
         * This method allows you to specify SequenceElement that will be used as UNK element, if UNK is used
         *
         * @param element
         * @return
         */
        @Override
        public Builder unknownElement(VocabWord element) {
            super.unknownElement(element);
            return this;
        }

        /**
         * This method allows you to specify, if UNK word should be used internally
         *
         * @param reallyUse
         * @return
         */
        @Override
        public Builder useUnknown(boolean reallyUse) {
            super.useUnknown(reallyUse);
            if (this.unknownElement == null) {
                this.unknownElement(new VocabWord(1.0, Glove.DEFAULT_UNK));
            }
            return this;
        }

        public Glove build() {
            presetTables();

            Glove ret = new Glove();


            // hardcoded value for glove

            if (sentenceIterator != null) {
                SentenceTransformer transformer = new SentenceTransformer.Builder()
                        .iterator(sentenceIterator)
                        .tokenizerFactory(tokenFactory)
                        .build();
                this.iterator = new AbstractSequenceIterator.Builder<VocabWord>(transformer).build();
            }


            ret.trainElementsVectors = true;
            ret.trainSequenceVectors = false;
            ret.useAdeGrad = true;
            this.useAdaGrad = true;

            ret.learningRate.set(this.learningRate);
            ret.resetModel = this.resetModel;
            ret.batchSize = this.batchSize;
            ret.iterator = this.iterator;
            ret.numEpochs = this.numEpochs;
            ret.numIterations = this.iterations;
            ret.layerSize = this.layerSize;

            ret.useUnknown = this.useUnknown;
            ret.unknownElement = this.unknownElement;



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

            ret.lookupTable = this.lookupTable;
            ret.vocab = this.vocabCache;
            ret.modelUtils = this.modelUtils;


            ret.elementsLearningAlgorithm = new GloVe.Builder<VocabWord>()
                    .learningRate(this.learningRate)
                    .shuffle(this.shuffle)
                    .symmetric(this.symmetric)
                    .xMax(this.xMax)
                    .alpha(this.alpha)
                    .maxMemory(maxmemory)
                    .build();

            return ret;
        }
    }
}
