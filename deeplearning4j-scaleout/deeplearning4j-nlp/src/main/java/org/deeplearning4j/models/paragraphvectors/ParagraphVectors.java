package org.deeplearning4j.models.paragraphvectors;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.deeplearning4j.models.abstractvectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.abstractvectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.abstractvectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareDocumentIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.documentiterator.interoperability.DocumentIteratorConverter;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.interoperability.SentenceIteratorConverter;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.util.Collection;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class ParagraphVectors extends Word2Vec {
    @Getter protected LabelsSource labelsSource;
    @Getter @Setter protected LabelAwareIterator labelAwareIterator;


    public static class Builder extends Word2Vec.Builder {
        protected LabelAwareIterator labelAwareIterator;
        protected LabelsSource labelsSource;
        protected DocumentIterator docIter;


        public Builder trainWordVectors(boolean trainElements) {
            this.trainElementsRepresentation(trainElements);
            return this;
        }

        public Builder labelsSource(@NonNull LabelsSource source) {
            this.labelsSource = source;
            return this;
        }

        @Deprecated
        public Builder labels(@NonNull List<String> labels) {

            return this;
        }

        public Builder iterate(@NonNull LabelAwareDocumentIterator iterator) {
            this.docIter = iterator;
            return this;
        }

        public Builder iterate(@NonNull LabelAwareSentenceIterator iterator) {
            this.sentenceIterator = iterator;
            return this;
        }

        public Builder iterate(@NonNull LabelAwareIterator iterator) {
            this.labelAwareIterator = iterator;
            return this;
        }

        @Override
        public Builder iterate(@NonNull DocumentIterator iterator) {
            this.docIter = iterator;
            return this;
        }

        @Override
        public Builder iterate(@NonNull SentenceIterator iterator) {
            this.sentenceIterator = iterator;
            return this;
        }

        @Override
        public ParagraphVectors build() {
            presetTables();

            ParagraphVectors ret = new ParagraphVectors();

            if (this.labelsSource == null) this.labelsSource = new LabelsSource();
            if (docIter != null) {
                /*
                        we're going to work with DocumentIterator.
                        First, we have to assume that user can provide LabelAwareIterator. In this case we'll use them, as provided source, and collec labels provided there
                        Otherwise we'll go for own labels via LabelsSource
                */

                if (docIter instanceof LabelAwareDocumentIterator) this.labelAwareIterator = new DocumentIteratorConverter((LabelAwareDocumentIterator) docIter, labelsSource);
                    else this.labelAwareIterator = new DocumentIteratorConverter(docIter, labelsSource);
            } else if (sentenceIterator != null) {
                    // we have SentenceIterator. Mechanics will be the same, as above
                 if (sentenceIterator instanceof LabelAwareSentenceIterator) this.labelAwareIterator = new SentenceIteratorConverter((LabelAwareSentenceIterator) sentenceIterator, labelsSource);
                      else this.labelAwareIterator = new SentenceIteratorConverter(sentenceIterator, labelsSource);
            } else if (labelAwareIterator != null) {
                 // if we have LabelAwareIterator defined, we have to be sure that LabelsSource is propagated properly
                 this.labelsSource = labelAwareIterator.getLabelsSource();
            } else  {
                // we have nothing, probably that's restored model building. ignore iterator for now.
                // probably there's few reasons to move iterator initialization code into ParagraphVectors methos. Like protected setLabelAwareIterator method.
                // TODO: to be investigated ^^^
                log.warn("Unexpected path taken");
                throw new RuntimeException("Unexpected path taken");
            }

            if (labelAwareIterator != null) {
                SentenceTransformer transformer = new SentenceTransformer.Builder()
                        .iterator(labelAwareIterator)
                        .tokenizerFactory(tokenizerFactory)
                        .build();
                this.iterator = new AbstractSequenceIterator.Builder<VocabWord>(transformer).build();
            } else throw new IllegalStateException("LabelAwareIterator is NULL");

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

            ret.lookupTable = this.lookupTable;

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


            // hardcoded to TRUE, since it's ParagraphVectors wrapper
            ret.trainElementsVectors = this.trainElementsVectors;
            ret.trainSequenceVectors = true;
            ret.labelsSource = this.labelsSource;
            ret.labelAwareIterator = this.labelAwareIterator;
            ret.iterator = this.iterator;

            return ret;
        }

        public Builder() {
            super();
        }

        public Builder(@NonNull VectorsConfiguration configuration) {
            super(configuration);
        }



        @Override
        public Builder tokenizerFactory(@NonNull TokenizerFactory tokenizerFactory) {
            super.tokenizerFactory(tokenizerFactory);
            return this;
        }

        @Override
        public Builder index(@NonNull InvertedIndex<VocabWord> index) {
            super.index(index);
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
            this.trainElementsVectors = trainElements;
            return this;
        }

        @Override
        public Builder trainSequencesRepresentation(boolean trainSequences) {
            this.trainSequenceVectors = trainSequences;
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
    }
}
