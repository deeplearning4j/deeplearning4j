package org.deeplearning4j.models.paragraphvectors;

import lombok.Getter;
import lombok.NonNull;
import org.deeplearning4j.models.abstractvectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareDocumentIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.util.Collection;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class ParagraphVectors extends Word2Vec {
    @Getter protected LabelsSource labelsSource;


    public static class Builder extends Word2Vec.Builder {


        public Builder trainWordVectors(boolean trainElements) {
            this.trainElementsRepresentation(trainElements);
            return this;
        }

        public Builder labelsSource(@NonNull LabelsSource source) {

            return this;
        }

        @Deprecated
        public Builder labels(@NonNull List<String> labels) {

            return this;
        }

        public Builder iterate(@NonNull LabelAwareDocumentIterator iterator) {

            return this;
        }

        public Builder iterate(@NonNull LabelAwareSentenceIterator iterator) {

            return this;
        }

        public Builder iterate(@NonNull LabelAwareIterator iterator) {

            return this;
        }

        @Override
        public Builder iterate(@NonNull DocumentIterator iterator) {
            super.iterate(iterator);
            return this;
        }

        @Override
        public Builder iterate(@NonNull SentenceIterator iterator) {
            super.iterate(iterator);
            return this;
        }

        @Override
        public ParagraphVectors build() {
            presetTables();

            ParagraphVectors ret = new ParagraphVectors();

            // hardcoded to TRUE, since it's ParagraphVectors wrapper
            ret.trainSequenceVectors = true;

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
            super.trainElementsRepresentation(trainElements);
            return this;
        }

        @Override
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
    }
}
