package org.deeplearning4j.models.paragraphvectors;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.*;
import org.deeplearning4j.text.documentiterator.interoperability.DocumentIteratorConverter;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.interoperability.SentenceIteratorConverter;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Basic ParagraphVectors (aka Doc2Vec) implementation for DL4j, as wrapper over SequenceVectors
 *
 * @author raver119@gmail.com
 */
public class ParagraphVectors extends Word2Vec {
    @Getter protected LabelsSource labelsSource;
    @Getter @Setter protected transient LabelAwareIterator labelAwareIterator;


    /**
     * This method takes raw text, applies tokenizer, and returns most probable label
     *
     * @param rawText
     * @return
     */
    public String predict(String rawText) {
        if (tokenizerFactory == null) throw new IllegalStateException("TokenizerFactory should be defined, prior to predict() call");

        List<String> tokens = tokenizerFactory.create(rawText).getTokens();
        List<VocabWord> document = new ArrayList<>();
        for (String token: tokens) {
            if (vocab.containsWord(token)) {
                document.add(vocab.wordFor(token));
            }
        }

        return predict(document);
    }

    /**
     * This method predicts label of the document.
     * Computes a similarity wrt the mean of the
     * representation of words in the document
     * @param document the document
     * @return the word distances for each label
     */
    public String predict(LabelledDocument document) {
        if (document.getReferencedContent() != null)
            return predict(document.getReferencedContent());
        else return predict(document.getContent());
    }

    /**
     * This method predicts label of the document.
     * Computes a similarity wrt the mean of the
     * representation of words in the document
     * @param document the document
     * @return the word distances for each label
     */
    public String predict(List<VocabWord> document) {
        /*
            This code was transferred from original ParagraphVectors DL4j implementation, and yet to be tested
         */
        if (document.isEmpty()) throw new IllegalStateException("Document has no words inside");

        INDArray arr = Nd4j.create(document.size(),this.layerSize);
        for(int i = 0; i < document.size(); i++) {
            arr.putRow(i,getWordVectorMatrix(document.get(i).getWord()));
        }

        INDArray docMean = arr.mean(0);
        Counter<String> distances = new Counter<>();

        for(String s : labelsSource.getLabels()) {
            INDArray otherVec = getWordVectorMatrix(s);
            double sim = Transforms.cosineSim(docMean, otherVec);
            distances.incrementCount(s, sim);
        }

        return distances.argMax();
    }

    /**
     * Predict several labels based on the document.
     * Computes a similarity wrt the mean of the
     * representation of words in the document
     * @param document raw text of the document
     * @return possible labels in descending order
     */
    public Collection<String> predictSeveral(@NonNull LabelledDocument document, int limit) {
        if (document.getReferencedContent() != null) {
            return predictSeveral(document.getReferencedContent(), limit);
        } else return predictSeveral(document.getContent(), limit);
    }

    /**
     * Predict several labels based on the document.
     * Computes a similarity wrt the mean of the
     * representation of words in the document
     * @param rawText raw text of the document
     * @return possible labels in descending order
     */
    public Collection<String> predictSeveral(String rawText, int limit) {
        if (tokenizerFactory == null) throw new IllegalStateException("TokenizerFactory should be defined, prior to predict() call");

        List<String> tokens = tokenizerFactory.create(rawText).getTokens();
        List<VocabWord> document = new ArrayList<>();
        for (String token: tokens) {
            if (vocab.containsWord(token)) {
                document.add(vocab.wordFor(token));
            }
        }

        return predictSeveral(document, limit);
    }

    /**
     * Predict several labels based on the document.
     * Computes a similarity wrt the mean of the
     * representation of words in the document
     * @param document the document
     * @return possible labels in descending order
     */
    public Collection<String> predictSeveral(List<VocabWord> document, int limit) {
        /*
            This code was transferred from original ParagraphVectors DL4j implementation, and yet to be tested
         */
        if (document.isEmpty()) throw new IllegalStateException("Document has no words inside");

        INDArray arr = Nd4j.create(document.size(),this.layerSize);
        for(int i = 0; i < document.size(); i++) {
            arr.putRow(i,getWordVectorMatrix(document.get(i).getWord()));
        }

        INDArray docMean = arr.mean(0);
        Counter<String> distances = new Counter<>();

        for(String s : labelsSource.getLabels()) {
            INDArray otherVec = getWordVectorMatrix(s);
            double sim = Transforms.cosineSim(docMean, otherVec);
            log.info("Similarity inside: ["+s+"] -> " + sim);
            distances.incrementCount(s, sim);
        }

        return distances.getSortedKeys().subList(0, limit);
    }


    public double similarityToLabel(String rawText, String label) {
        if (tokenizerFactory == null) throw new IllegalStateException("TokenizerFactory should be defined, prior to predict() call");

        List<String> tokens = tokenizerFactory.create(rawText).getTokens();
        List<VocabWord> document = new ArrayList<>();
        for (String token: tokens) {
            if (vocab.containsWord(token)) {
                document.add(vocab.wordFor(token));
            }
        }
        return similarityToLabel(document, label);
    }

    public double similarityToLabel(LabelledDocument document, String label) {
        if (document.getReferencedContent() != null) {
            return similarityToLabel(document.getReferencedContent(), label);
        } else return similarityToLabel(document.getContent(), label);
    }

    public double similarityToLabel(List<VocabWord> document, String label) {
        if (document.isEmpty()) throw new IllegalStateException("Document has no words inside");

        INDArray arr = Nd4j.create(document.size(),this.layerSize);
        for(int i = 0; i < document.size(); i++) {
            arr.putRow(i,getWordVectorMatrix(document.get(i).getWord()));
        }

        INDArray docMean = arr.mean(0);

        INDArray otherVec = getWordVectorMatrix(label);
        double sim = Transforms.cosineSim(docMean, otherVec);
        return sim;
    }



    public static class Builder extends Word2Vec.Builder {
        protected LabelAwareIterator labelAwareIterator;
        protected LabelsSource labelsSource;
        protected DocumentIterator docIter;




        /**
         * This method allows you to use pre-built WordVectors model (Word2Vec or GloVe) for ParagraphVectors.
         * Existing model will be transferred into new model before training starts.
         *
         * PLEASE NOTE: Non-normalized model is recommended to use here.
         *
         * @param vec existing WordVectors model
         * @return
         */
        @Override
        protected Builder useExistingWordVectors(@NonNull WordVectors vec) {
            this.existingVectors = vec;
            return this;
        }

        /**
         * This method defines, if words representations should be build together with documents representations.
         *
         * @param trainElements
         * @return
         */
        public Builder trainWordVectors(boolean trainElements) {
            this.trainElementsRepresentation(trainElements);
            return this;
        }

        /**
         * This method attaches pre-defined labels source to ParagraphVectors
         *
         * @param source
         * @return
         */
        public Builder labelsSource(@NonNull LabelsSource source) {
            this.labelsSource = source;
            return this;
        }

        /**
         * This method builds new LabelSource instance from labels.
         *
         * PLEASE NOTE: Order synchro between labels and input documents delegated to end-user.
         * PLEASE NOTE: Due to order issues it's recommended to use label aware iterators instead.
         *
         * @param labels
         * @return
         */
        @Deprecated
        public Builder labels(@NonNull List<String> labels) {
            this.labelsSource = new LabelsSource(labels);
            return this;
        }

        /**
         * This method used to feed LabelAwareDocumentIterator, that contains training corpus, into ParagraphVectors
         *
         * @param iterator
         * @return
         */
        public Builder iterate(@NonNull LabelAwareDocumentIterator iterator) {
            this.docIter = iterator;
            return this;
        }

        /**
         * This method used to feed LabelAwareSentenceIterator, that contains training corpus, into ParagraphVectors
         *
         * @param iterator
         * @return
         */
        public Builder iterate(@NonNull LabelAwareSentenceIterator iterator) {
            this.sentenceIterator = iterator;
            return this;
        }

        /**
         * This method used to feed LabelAwareIterator, that contains training corpus, into ParagraphVectors
         *
         * @param iterator
         * @return
         */
        public Builder iterate(@NonNull LabelAwareIterator iterator) {
            this.labelAwareIterator = iterator;
            return this;
        }

        /**
         * This method used to feed DocumentIterator, that contains training corpus, into ParagraphVectors
         *
         * @param iterator
         * @return
         */
        @Override
        public Builder iterate(@NonNull DocumentIterator iterator) {
            this.docIter = iterator;
            return this;
        }

        /**
         * This method used to feed SentenceIterator, that contains training corpus, into ParagraphVectors
         *
         * @param iterator
         * @return
         */
        @Override
        public Builder iterate(@NonNull SentenceIterator iterator) {
            this.sentenceIterator = iterator;
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

        @Override
        public ParagraphVectors build() {
            presetTables();

            ParagraphVectors ret = new ParagraphVectors();

            if (this.existingVectors != null) {
                this.trainElementsVectors = false;
                this.elementsLearningAlgorithm = null;
            }

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
            }

            if (labelAwareIterator != null) {
                SentenceTransformer transformer = new SentenceTransformer.Builder()
                        .iterator(labelAwareIterator)
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

            ret.trainElementsVectors = this.trainElementsVectors;
            ret.trainSequenceVectors = this.trainSequenceVectors;

            ret.elementsLearningAlgorithm = this.elementsLearningAlgorithm;
            ret.sequenceLearningAlgorithm = this.sequenceLearningAlgorithm;

            ret.tokenizerFactory = this.tokenizerFactory;

            ret.existingModel = this.existingVectors;

            ret.lookupTable = this.lookupTable;
            ret.modelUtils = this.modelUtils;

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


        /**
         * This method defines TokenizerFactory to be used for strings tokenization during training
         * PLEASE NOTE: If external VocabCache is used, the same TokenizerFactory should be used to keep derived tokens equal.
         *
         * @param tokenizerFactory
         * @return
         */
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
         * @param stopList
         * @return
         */
        @Override
        public Builder stopWords(@NonNull List<String> stopList) {
            super.stopWords(stopList);
            return this;
        }

        /**
         * This method defines, if words representation should be build together with documents representations.
         *
         * @param trainElements
         * @return
         */
        @Override
        public Builder trainElementsRepresentation(boolean trainElements) {
            this.trainElementsVectors = trainElements;
            return this;
        }

        /**
         * This method is hardcoded to TRUE, since that's whole point of ParagraphVectors
         *
         * @param trainSequences
         * @return
         */
        @Override
        public Builder trainSequencesRepresentation(boolean trainSequences) {
            this.trainSequenceVectors = trainSequences;
            return this;
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
         * This method defines random seed for random numbers generator
         * @param randomSeed
         * @return
         */
        @Override
        public Builder seed(long randomSeed) {
            super.seed(randomSeed);
            return this;
        }
    }
}
