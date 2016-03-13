package org.deeplearning4j.models.sequencevectors;

import lombok.Getter;
import lombok.NonNull;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.learning.SequenceLearningAlgorithm;
import org.deeplearning4j.models.embeddings.learning.impl.sequence.DBOW;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.nd4j.linalg.heartbeat.Heartbeat;
import org.nd4j.linalg.heartbeat.reports.Environment;
import org.nd4j.linalg.heartbeat.reports.Event;
import org.nd4j.linalg.heartbeat.reports.Task;
import org.nd4j.linalg.heartbeat.utils.EnvironmentUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * SequenceVectors implements abstract features extraction for Sequences and SequenceElements, using SkipGram, CBOW or DBOW (for Sequence features extraction).
 *
 *
 * @author raver119@gmail.com
 */
public class SequenceVectors<T extends SequenceElement> extends WordVectorsImpl<T> implements WordVectors {
    protected transient SequenceIterator<T> iterator;

    protected transient ElementsLearningAlgorithm<T> elementsLearningAlgorithm;
    protected transient SequenceLearningAlgorithm<T> sequenceLearningAlgorithm;

    @Getter protected VectorsConfiguration configuration;

    protected static final Logger log = LoggerFactory.getLogger(SequenceVectors.class);

    protected transient WordVectors existingModel;
    protected transient T unknownElement;

    public interface EpochListener {
        void epochCompleted(int epoch, WordVectors wordVectors);
    }

    @Setter protected EpochListener epochListener;

    /**
     * Builds vocabulary from provided SequenceIterator instance
     */
    public void buildVocab() {


        VocabConstructor<T> constructor = new VocabConstructor.Builder<T>()
                .addSource(iterator, minWordFrequency)
                .setTargetVocabCache(vocab)
                .fetchLabels(trainSequenceVectors)
                .setStopWords(stopWords)
                .build();

        if (existingModel != null && lookupTable instanceof InMemoryLookupTable && existingModel.lookupTable() instanceof InMemoryLookupTable) {
            log.info("Merging existing vocabulary into the current one...");
            /*
                if we have existing model defined, we're forced to fetch labels only.
                the rest of vocabulary & weights should be transferred from existing model
             */

            constructor.buildMergedVocabulary(existingModel, true);

            /*
                Now we have vocab transferred, and we should transfer syn0 values into lookup table
             */
            ((InMemoryLookupTable<VocabWord>) lookupTable).consume((InMemoryLookupTable<VocabWord>) existingModel.lookupTable());
        } else {
            log.info("Starting vocabulary building...");
            // if we don't have existing model defined, we just build vocabulary


            constructor.buildJointVocabulary(false, true);

            if (useUnknown && unknownElement != null && !vocab.containsWord(unknownElement.getLabel())) {
                log.info("Adding UNK element...");
                unknownElement.setSpecial(true);
                unknownElement.markAsLabel(false);
                unknownElement.setIndex(vocab.numWords());
                vocab.addToken(unknownElement);

            }


            // check for malformed inputs. if numWords/numSentences ratio is huge, then user is passing something weird
            if (vocab.numWords() / constructor.getNumberOfSequences() > 1000) {
                log.warn("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                log.warn("!                                                                                      !");
                log.warn("! Your input looks malformed: number of sentences is too low, model accuracy may suffer!");
                log.warn("!                                                                                      !");
                log.warn("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            }
        }


    }



    /**
     * Starts training over
     */
    public void fit() {
        if (!trainElementsVectors && !trainSequenceVectors) throw new IllegalStateException("You should define at least one training goal 'trainElementsRepresentation' or 'trainSequenceRepresentation'");
        if (iterator == null) throw new IllegalStateException("You can't fit() data without SequenceIterator defined");

        if (resetModel || (lookupTable != null && vocab != null && vocab.numWords() == 0)) {
            // build vocabulary from scratches
            buildVocab();
        }

        if (vocab == null || lookupTable == null || vocab.numWords() == 0) throw new IllegalStateException("You can't fit() model with empty Vocabulary or WeightLookupTable");

        // if model vocab and lookupTable is built externally we basically should check that lookupTable was properly initialized
        if (!resetModel || existingModel != null) {
            lookupTable.resetWeights(false);
        } else {
            // otherwise we reset weights, independent of actual current state of lookup table
            lookupTable.resetWeights(true);
        }

        log.info("Building learning algorithms:");
        if (trainElementsVectors && elementsLearningAlgorithm != null) {
            log.info("          building ElementsLearningAlgorithm: [" +elementsLearningAlgorithm.getCodeName()+ "]");
            elementsLearningAlgorithm.configure(vocab, lookupTable, configuration);
            elementsLearningAlgorithm.pretrain(iterator);
        }
        if (trainSequenceVectors && sequenceLearningAlgorithm != null) {
            log.info("          building SequenceLearningAlgorithm: [" +sequenceLearningAlgorithm.getCodeName()+ "]");
            sequenceLearningAlgorithm.configure(vocab, lookupTable, configuration);
            sequenceLearningAlgorithm.pretrain(this.iterator);
        }

        log.info("Starting learning process...");
        if (this.stopWords == null) this.stopWords = new ArrayList<>();
        for (int currentEpoch = 1; currentEpoch <= numEpochs; currentEpoch++) {
            final AtomicLong linesCounter = new AtomicLong(0);
            final AtomicLong wordsCounter = new AtomicLong(0);

            AsyncSequencer sequencer = new AsyncSequencer(this.iterator, this.stopWords);
            sequencer.start();


            //final VectorCalculationsThread[] threads = new VectorCalculationsThread[workers];
            final List<VectorCalculationsThread> threads = new ArrayList<>();
            for (int x = 0; x < workers; x++) {
                threads.add(x, new VectorCalculationsThread(x, currentEpoch, wordsCounter, vocab.totalWordOccurrences(), linesCounter,  sequencer));
                threads.get(x).start();
            }

            try {
                sequencer.join();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            for (int x = 0; x < workers; x++) {
                try {
                    threads.get(x).join();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }

            // TODO: fix this to non-exclusive termination
            if (trainElementsVectors && elementsLearningAlgorithm != null) {
                if (!trainSequenceVectors || sequenceLearningAlgorithm == null) {
                    if (elementsLearningAlgorithm.isEarlyTerminationHit()) break;
                }
            }

            if (trainSequenceVectors && sequenceLearningAlgorithm!= null) {
                if (!trainElementsVectors || elementsLearningAlgorithm == null) {
                    if (sequenceLearningAlgorithm.isEarlyTerminationHit()) break;
                }
            }
            log.info("Epoch: [" + currentEpoch+ "]; Words vectorized so far: [" + wordsCounter.get() + "];  Lines vectorized so far: [" + linesCounter.get() + "]; learningRate: [" + minLearningRate + "]");

            if (epochListener != null) {
                epochListener.epochCompleted(currentEpoch, this);
            }
        }
    }


    protected void trainSequence(@NonNull Sequence<T> sequence, AtomicLong nextRandom, double alpha) {

        if (sequence.getElements().size() == 0) return;

        if (trainElementsVectors) {
            // call for ElementsLearningAlgorithm
            nextRandom.set(nextRandom.get() * 25214903917L + 11);
            if (!elementsLearningAlgorithm.isEarlyTerminationHit()) elementsLearningAlgorithm.learnSequence(sequence, nextRandom, alpha);
        }

        if (trainSequenceVectors)  {
            // call for SequenceLearningAlgorithm
            nextRandom.set(nextRandom.get() * 25214903917L + 11);
            //dbow(i, sequence, (int) nextRandom.get() % window, nextRandom, alpha);
            if (!sequenceLearningAlgorithm.isEarlyTerminationHit()) sequenceLearningAlgorithm.learnSequence(sequence, nextRandom, alpha);
        }
    }


    public static class Builder<T extends SequenceElement> {
        protected VocabCache<T> vocabCache;
        protected WeightLookupTable<T> lookupTable;
        protected SequenceIterator<T> iterator;
        protected ModelUtils<T> modelUtils = new BasicModelUtils<>();

        protected WordVectors existingVectors;

        protected double sampling = 0;
        protected double negative = 0;
        protected double learningRate = 0.025;
        protected double minLearningRate = 0.0001;
        protected int minWordFrequency = 0;
        protected int iterations = 1;
        protected int numEpochs = 1;
        protected int layerSize = 100;
        protected int window = 5;
        protected boolean hugeModelExpected = false;
        protected int batchSize = 100;
        protected int learningRateDecayWords;
        protected long seed;
        protected boolean useAdaGrad = false;
        protected boolean resetModel = true;
        protected int workers = Runtime.getRuntime().availableProcessors();
        protected boolean useUnknown = false;

        protected boolean trainSequenceVectors = false;
        protected boolean trainElementsVectors = true;

        protected List<String> stopWords = new ArrayList<>();

        protected VectorsConfiguration configuration = new VectorsConfiguration();

        protected transient T unknownElement;
        protected String UNK = configuration.getUNK();
        protected String STOP = configuration.getSTOP();

        // defaults values for learning algorithms are set here
        protected ElementsLearningAlgorithm<T> elementsLearningAlgorithm = new SkipGram<T>();
        protected SequenceLearningAlgorithm<T> sequenceLearningAlgorithm = new DBOW<T>();

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
            this.UNK = configuration.getUNK();
            this.STOP = configuration.getSTOP();

            if (configuration.getElementsLearningAlgorithm() != null && !configuration.getElementsLearningAlgorithm().isEmpty()) {
                this.elementsLearningAlgorithm(configuration.getElementsLearningAlgorithm());
            }

            if (configuration.getSequenceLearningAlgorithm() != null && !configuration.getSequenceLearningAlgorithm().isEmpty()) {
                this.sequenceLearningAlgorithm(configuration.getSequenceLearningAlgorithm());
            }

            if (configuration.getStopList() != null) this.stopWords.addAll(configuration.getStopList());
        }

        /**
         * This method allows you to use pre-built WordVectors model (SkipGram or GloVe) for DBOW sequence learning.
         * Existing model will be transferred into new model before training starts.
         *
         * PLEASE NOTE: This model has no effect for elements learning algorithms. Only sequence learning is affected.
         * PLEASE NOTE: Non-normalized model is recommended to use here.
         *
         * @param vec existing WordVectors model
         * @return
         */
        protected Builder<T> useExistingWordVectors(@NonNull WordVectors vec) {
            this.existingVectors = vec;
            return this;
        }

        /**
         * This method defines SequenceIterator to be used for model building
         * @param iterator
         * @return
         */
        public Builder<T> iterate(@NonNull SequenceIterator<T> iterator) {
            this.iterator = iterator;
            return this;
        }

        /**
         * Sets specific LearningAlgorithm as Sequence Learning Algorithm
         *
         * @param algoName fully qualified class name
         * @return
         */
        public Builder<T> sequenceLearningAlgorithm(@NonNull String algoName) {
            try {
                Class clazz = Class.forName(algoName);
                sequenceLearningAlgorithm = (SequenceLearningAlgorithm<T>) clazz.newInstance();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            return this;
        }

        /**
         * Sets specific LearningAlgorithm as Sequence Learning Algorithm
         *
         * @param algorithm SequenceLearningAlgorithm implementation
         * @return
         */
        public Builder<T> sequenceLearningAlgorithm(@NonNull SequenceLearningAlgorithm<T> algorithm) {
            this.sequenceLearningAlgorithm = algorithm;
            return this;
        }

        /**
         * * Sets specific LearningAlgorithm as Elements Learning Algorithm
         *
         * @param algoName fully qualified class name
         * @return
         */
        public Builder<T> elementsLearningAlgorithm(@NonNull String algoName) {
            try {
                Class clazz = Class.forName(algoName);
                elementsLearningAlgorithm = (ElementsLearningAlgorithm<T>) clazz.newInstance();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            return this;
        }

        /**
         * * Sets specific LearningAlgorithm as Elements Learning Algorithm
         *
         * @param algorithm ElementsLearningAlgorithm implementation
         * @return
         */
        public Builder<T> elementsLearningAlgorithm(@NonNull ElementsLearningAlgorithm<T> algorithm) {
            this.elementsLearningAlgorithm = algorithm;
            return this;
        }

        /**
         * This method defines batchSize option, viable only if iterations > 1
         *
         * @param batchSize
         * @return
         */
        public Builder<T> batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        /**
         * This method defines how much iterations should be done over batched sequences.
         *
         * @param iterations
         * @return
         */
        public Builder<T> iterations(int iterations) {
            this.iterations = iterations;
            return this;
        }

        /**
         * This method defines how much iterations should be done over whole training corpus during modelling
         * @param numEpochs
         * @return
         */
        public Builder<T> epochs(int numEpochs) {
            this.numEpochs = numEpochs;
            return this;
        }

        /**
         * Sets number of worker threads to be used in calculations
         *
         * @param numWorkers
         * @return
         */
        public Builder<T> workers(int numWorkers) {
            this.workers = numWorkers;
            return this;
        }

        /**
         * This method defines if Adaptive Gradients should be used in calculations
         *
         * @param reallyUse
         * @return
         */
        public Builder<T> useAdaGrad(boolean reallyUse) {
            this.useAdaGrad = reallyUse;
            return this;
        }

        /**
         * This method defines number of dimensions for outcome vectors.
         * Please note: This option has effect only if lookupTable wasn't defined during building process.
         *
         * @param layerSize
         * @return
         */
        public Builder<T> layerSize(int layerSize) {
            this.layerSize = layerSize;
            return this;
        }

        /**
         * This method defines initial learning rate.
         * Default value is 0.025
         *
         * @param learningRate
         * @return
         */
        public Builder<T> learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        /**
         * This method defines minimal element frequency for elements found in the training corpus. All elements with frequency below this threshold will be removed before training.
         * Please note: this method has effect only if vocabulary is built internally.
         *
         * @param minWordFrequency
         * @return
         */
        public Builder<T> minWordFrequency(int minWordFrequency) {
            this.minWordFrequency = minWordFrequency;
            return this;
        }

        /**
         * This method defines minimum learning rate after decay being applied.
         * Default value is 0.01
         *
         * @param minLearningRate
         * @return
         */
        public Builder<T> minLearningRate(double minLearningRate) {
            this.minLearningRate = minLearningRate;
            return this;
        }

        /**
         * This method defines, should all model be reset before training. If set to true, vocabulary and WeightLookupTable will be reset before training, and will be built from scratches
         *
         * @param reallyReset
         * @return
         */
        public Builder<T> resetModel(boolean reallyReset) {
            this.resetModel = reallyReset;
            return this;
        }

        /**
         * You can pass externally built vocabCache object, containing vocabulary
         *
         * @param vocabCache
         * @return
         */
        public Builder<T> vocabCache(@NonNull VocabCache<T> vocabCache) {
            this.vocabCache = vocabCache;
            return this;
        }

        /**
         * You can pass externally built WeightLookupTable, containing model weights and vocabulary.
         *
         * @param lookupTable
         * @return
         */
        public Builder<T> lookupTable(@NonNull WeightLookupTable<T> lookupTable) {
            this.lookupTable = lookupTable;
            return this;
        }

        /**
         * This method defines sub-sampling threshold.
         *
         * @param sampling
         * @return
         */
        public Builder<T> sampling(double sampling) {
            this.sampling = sampling;
            return this;
        }

        /**
         * This method defines negative sampling value for skip-gram algorithm.
         *
         * @param negative
         * @return
         */
        public Builder<T> negativeSample(double negative) {
            this.negative = negative;
            return this;
        }

        /**
         *  You can provide collection of objects to be ignored, and excluded out of model
         *  Please note: Object labels and hashCode will be used for filtering
         *
         * @param stopList
         * @return
         */
        public Builder<T> stopWords(@NonNull List<String> stopList) {
            this.stopWords.addAll(stopList);
            return this;
        }

        /**
         *
         * @param trainElements
         * @return
         */
        public Builder<T> trainElementsRepresentation(boolean trainElements) {
            this.trainElementsVectors = trainElements;
            return this;
        }

        public Builder<T> trainSequencesRepresentation(boolean trainSequences) {
            this.trainSequenceVectors = trainSequences;
            return this;
        }

        /**
         * You can provide collection of objects to be ignored, and excluded out of model
         * Please note: Object labels and hashCode will be used for filtering
         *
         * @param stopList
         * @return
         */
        public Builder<T> stopWords(@NonNull Collection<T> stopList) {
            for (T word: stopList) {
                this.stopWords.add(word.getLabel());
            }
            return this;
        }

        /**
         * Sets window size for skip-Gram training
         *
         * @param windowSize
         * @return
         */
        public Builder<T> windowSize(int windowSize) {
            this.window = windowSize;
            return this;
        }

        /**
         * Sets seed for random numbers generator.
         * Please note: this has effect only if vocabulary and WeightLookupTable is built internally
         *
         * @param randomSeed
         * @return
         */
        public Builder<T> seed(long randomSeed) {
            // has no effect in original w2v actually
            return this;
        }

        /**
         * ModelUtils implementation, that will be used to access model.
         * Methods like: similarity, wordsNearest, accuracy are provided by user-defined ModelUtils
         *
         * @param modelUtils model utils to be used
         * @return
         */
        public Builder<T> modelUtils(@NonNull ModelUtils<T> modelUtils) {
            this.modelUtils = modelUtils;
            return this;
        }

        /**
         * This method allows you to specify, if UNK word should be used internally
         * @param reallyUse
         * @return
         */
        public Builder<T> useUnknown(boolean reallyUse) {
            this.useUnknown = reallyUse;
            return this;
        }

        /**
         * This method allows you to specify SequenceElement that will be used as UNK element, if UNK is used
         * @param element
         * @return
         */
        public Builder<T> unknownElement(@NonNull T element) {
            this.unknownElement = element;
            this.UNK = element.getLabel();
            return this;
        }


        /**
         * This method creates new WeightLookupTable<T> and VocabCache<T> if there were none set
         */
        protected void presetTables() {
            if (lookupTable == null) {

                if (vocabCache == null) {
                    vocabCache = new AbstractCache.Builder<T>()
                            .hugeModelExpected(hugeModelExpected)
                            .scavengerRetentionDelay(this.configuration.getScavengerRetentionDelay())
                            .scavengerThreshold(this.configuration.getScavengerActivationThreshold())
                            .minElementFrequency(minWordFrequency)
                            .build();
                }

                lookupTable = new InMemoryLookupTable.Builder<T>()
                        .useAdaGrad(this.useAdaGrad)
                        .cache(vocabCache)
                        .negative(negative)
                        .vectorLength(layerSize)
                        .lr(learningRate)
                        .seed(seed)
                        .build();
            }

            if (trainElementsVectors) {
                if (elementsLearningAlgorithm == null) {
                    // create default implementation of ElementsLearningAlgorithm
                    elementsLearningAlgorithm = new SkipGram<>();
                }
            }

            if (trainSequenceVectors) {
                if (sequenceLearningAlgorithm == null) {
                    // create default implementation of SequenceLearningAlgorithm
                    sequenceLearningAlgorithm = new DBOW<>();
                }
            }

            this.modelUtils.init(lookupTable);
        }

        /**
         * Build SequenceVectors instance with defined settings/options
         * @return
         */
        public SequenceVectors<T> build() {
            presetTables();

            SequenceVectors<T> vectors = new SequenceVectors<>();

            if (this.existingVectors != null) {
                this.trainElementsVectors = false;
                this.elementsLearningAlgorithm = null;
            }

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
            vectors.workers = this.workers;

            vectors.iterator = this.iterator;
            vectors.lookupTable = this.lookupTable;
            vectors.modelUtils = this.modelUtils;
            vectors.useUnknown = this.useUnknown;
            vectors.unknownElement = this.unknownElement;


            vectors.trainElementsVectors = this.trainElementsVectors;
            vectors.trainSequenceVectors = this.trainSequenceVectors;

            vectors.elementsLearningAlgorithm = this.elementsLearningAlgorithm;
            vectors.sequenceLearningAlgorithm = this.sequenceLearningAlgorithm;

            vectors.existingModel = this.existingVectors;

            vectors.setUNK(this.UNK);

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
            this.configuration.setUNK(this.UNK);

            vectors.configuration = this.configuration;

            return vectors;
        }
    }

    /**
     * This class is used to fetch data from iterator in background thread, and convert it to List<VocabularyWord>
     *
     * It becomes very usefull if text processing pipeline behind iterator is complex, and we're not loading data from simple text file with whitespaces as separator.
     * Since this method allows you to hide preprocessing latency in background.
     *
     * This mechanics will be change to PrefetchingSentenceIterator wrapper.
     */
    protected class AsyncSequencer extends Thread implements Runnable {
        private final SequenceIterator<T> iterator;
        private final LinkedBlockingQueue<Sequence<T>> buffer;
   //     private final AtomicLong linesCounter;
        private final int limitUpper = 10000;
        private final int limitLower = 5000;
        private AtomicBoolean isRunning = new AtomicBoolean(true);
        private AtomicLong nextRandom;
        private List<String> stopList;

        public AsyncSequencer(SequenceIterator<T> iterator, @NonNull List<String> stopList) {
            this.iterator = iterator;
            this.buffer = new LinkedBlockingQueue<>();
//            this.linesCounter = linesCounter;
            this.setName("AsyncSequencer thread");
            this.nextRandom = new AtomicLong(workers + 1);
            this.iterator.reset();
            this.stopList = stopList;
        }

        @Override
        public void run() {
            isRunning.set(true);
            while (this.iterator.hasMoreSequences()) {

                // if buffered level is below limitLower, we're going to fetch limitUpper number of strings from fetcher
                if (buffer.size() < limitLower) {
                    update();
                    AtomicInteger linesLoaded = new AtomicInteger(0);
                    while (linesLoaded.getAndIncrement() < limitUpper && this.iterator.hasMoreSequences() ) {
                        Sequence<T> document = this.iterator.nextSequence();

                        /*
                            We can't hope/assume that underlying iterator contains synchronized elements
                            That's why we're going to rebuild sequence from vocabulary
                          */
                        Sequence<T> newSequence = new Sequence<>();

                        if (document.getSequenceLabel() != null) {
                            T newLabel = vocab.wordFor(document.getSequenceLabel().getLabel());
                            if (newLabel != null) newSequence.setSequenceLabel(newLabel);
                        }

                        for (T element: document.getElements()) {
                            if (stopList.contains(element.getLabel())) continue;
                            T realElement = vocab.wordFor(element.getLabel());

                            // please note: this serquence element CAN be absent in vocab, due to minFreq or stopWord or whatever else
                            if (realElement != null) {
                                newSequence.addElement(realElement);
                            } else if (useUnknown && unknownElement != null) {
                                newSequence.addElement(unknownElement);
                            }
                        }

                        // due to subsampling and null words, new sequence size CAN be 0, so there's no need to insert empty sequence into processing chain
                        if (newSequence.getElements().size() > 0) buffer.add(newSequence);

                        linesLoaded.incrementAndGet();
                    }
                } else {
                    try {
                        Thread.sleep(50);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
            isRunning.set(false);
        }

        public boolean hasMoreLines() {
            // statement order does matter here, since there's possible race condition
            return buffer.size() > 0 || isRunning.get();
        }

        public Sequence<T> nextSentence() {
            try {
                return buffer.poll(3L, TimeUnit.SECONDS);
            } catch (Exception e) {
                return null;
            }
        }
    }

    /**
     * VectorCalculationsThreads are used for vector calculations, and work together with AsyncIteratorDigitizer.
     * Basically, all they do is just transfer of digitized sentences into math layer.
     *
     * Please note, they do not iterate the sentences over and over, each sentence processed only once.
     * Training corpus iteration is implemented in fit() method.
     *
     */
    private class VectorCalculationsThread extends Thread implements Runnable {
        private final int threadId;
        private final int epochNumber;
        private final AtomicLong wordsCounter;
        private final long totalWordsCount;
        private final AtomicLong totalLines;

        private final AsyncSequencer digitizer;
        private final AtomicLong nextRandom;

        /*
                Long constructors suck, so this should be reduced to something reasonable later
         */
        public VectorCalculationsThread(int threadId, int epoch, AtomicLong wordsCounter, long totalWordsCount, AtomicLong linesCounter, AsyncSequencer digitizer) {
            this.threadId = threadId;
            this.epochNumber = epoch;
            this.wordsCounter = wordsCounter;
            this.totalWordsCount = totalWordsCount;
            this.totalLines = linesCounter;
            this.digitizer = digitizer;
            this.nextRandom = new AtomicLong(this.threadId);
            this.setName("VectorCalculationsThread " + this.threadId);
        }

        @Override
        public void run() {
            while ( digitizer.hasMoreLines()) {
                try {
                    // get current sentence as list of VocabularyWords
                    List<Sequence<T>> sequences = new ArrayList<>();
                    for (int x = 0; x < batchSize; x++) {
                        if (digitizer.hasMoreLines()) {
                            Sequence<T> sequence = digitizer.nextSentence();
                            if (sequence != null) {
                                sequences.add(sequence);
                            }
                        }
                    }
                    /*
                            TODO: investigate, if fix needed here to become iteration-dependent, not line-position
                      */
                    double alpha = 0.025;

                    if (sequences.size() == 0) {
                        continue;
                    }

                    // getting back number of iterations
                    for (int i = 0; i < numIterations; i++) {
                        for (int x = 0; x< sequences.size(); x++) {
                            Sequence<T> sequence = sequences.get(x);

                            alpha = Math.max(minLearningRate, learningRate.get() * (1 - (1.0 * this.wordsCounter.get() / (double) this.totalWordsCount)));

                            trainSequence(sequence, nextRandom, alpha);

                            // increment processed word count, please note: this affects learningRate decay
                            totalLines.incrementAndGet();
                            this.wordsCounter.addAndGet(sequence.getElements().size());

                            if (totalLines.get() % 100000 == 0) log.info("Epoch: [" + this.epochNumber+ "]; Words vectorized so far: [" + this.wordsCounter.get() + "];  Lines vectorized so far: [" + this.totalLines.get() + "]; learningRate: [" + alpha + "]");
                        }
                    }


                } catch (Exception  e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }
}
