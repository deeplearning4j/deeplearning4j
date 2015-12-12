package org.deeplearning4j.models.abstractvectors;

import lombok.NonNull;
import lombok.Setter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.graph.iterator.GraphWalkIterator;
import org.deeplearning4j.models.abstractvectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.abstractvectors.sequence.Sequence;
import org.deeplearning4j.models.abstractvectors.sequence.SequenceElement;
import org.deeplearning4j.models.abstractvectors.transformers.SequenceTransformer;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * AbstractVectors implements abstract features extraction for Sequences and SequenceElements, using SkipGram, CBOW or DBOW (for Sequence features extraction).
 *
 * DO NOT USE, IT'S JUST A DRAFT FOR FUTURE WordVectorsImpl changes
 * @author raver119@gmail.com
 */
public class AbstractVectors<T extends SequenceElement> extends WordVectorsImpl<T> implements WordVectors {
    protected SequenceIterator<T> iterator;

    protected VectorsConfiguration configuration;

    protected static final Logger log = LoggerFactory.getLogger(AbstractVectors.class);

    /**
     * Starts training over
     */
    public void fit() {
        if (iterator == null) throw new IllegalStateException("You can't fit() data without SequenceIterator defined");

        if (resetModel) {
            // build vocabulary from scratches
        }

        log.info("Starting learning process...");
        for (int currentEpoch = 0; currentEpoch < numEpochs; currentEpoch++) {
            AtomicLong linesCounter = new AtomicLong(0);
            AtomicLong wordsCounter = new AtomicLong(0);

            AsyncSequencer sequencer = new AsyncSequencer(this.iterator);
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
        }
    }


    /**
     * Train the distributed bag of words
     * model
     * @param i the word to train
     * @param sequence of elements with labels to train over
     * @param b
     * @param nextRandom
     * @param alpha
     */
    public void dbow(int i, Sequence<T> sequence, int b, AtomicLong nextRandom, double alpha) {

        final T word = sequence.getElements().get(i);
        List<T> sentence = sequence.getElements();

        // TODO: fix this, there should be option to have few labels per sequence
        List<T> labels = new ArrayList<>(); //(List<T>) sequence.getSequenceLabel();
        labels.add(sequence.getSequenceLabel());
        //    final VocabWord word = labels.get(0);

        if(word == null || sentence.isEmpty())
            return;


        int end =  window * 2 + 1 - b;
        for(int a = b; a < end; a++) {
            if(a != window) {
                int c = i - window + a;
                if(c >= 0 && c < labels.size()) {
                    T lastWord = labels.get(c);
                    iterate(word, lastWord,nextRandom,alpha);
                }
            }
        }
    }

    /**
     * Train via skip gram
     * @param i
     * @param sentence
     */
    protected void skipGram(int i,List<T> sentence, int b,AtomicLong nextRandom,double alpha) {

        final T word = sentence.get(i);
        if(word == null || sentence.isEmpty())
            return;

        int end =  window * 2 + 1 - b;
        for(int a = b; a < end; a++) {
            if(a != window) {
                int c = i - window + a;
                if(c >= 0 && c < sentence.size()) {
                    T lastWord = sentence.get(c);
                    iterate(word,lastWord,nextRandom,alpha);
                }
            }
        }
    }

    /**
     * Train the word vector
     * on the given words
     * @param w1 the first word to fit
     */
    protected void  iterate(T w1, T w2,AtomicLong nextRandom,double alpha) {
        lookupTable.iterateSample(w1,w2,nextRandom,alpha);

    }

    protected void trainSequence(Sequence<T> sequence, AtomicLong nextRandom, double alpha) {

        if (trainWordVectors) for(int i = 0; i < sequence.getElements().size(); i++) {
            nextRandom.set(nextRandom.get() * 25214903917L + 11);
            skipGram(i, sequence.getElements(), (int) nextRandom.get() % window,nextRandom,alpha);
        }

        if (trainSequenceVectors) for(int i = 0; i < sequence.getElements().size(); i++) {
            nextRandom.set(nextRandom.get() * 25214903917L + 11);
            dbow(i, sequence, (int) nextRandom.get() % window, nextRandom, alpha);
        }
    }

    public static class Builder<T extends SequenceElement> {
        protected VocabCache<T> vocabCache;
        protected WeightLookupTable<T> lookupTable;
        protected SequenceIterator<T> iterator;

        protected double sampling;
        protected double negative;
        protected double learningRate;
        protected double minLearningRate;
        protected int minWordFrequency;
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

        public Builder<T> iterate(@NonNull SequenceIterator<T> iterator) {
            this.iterator = iterator;
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

        public Builder<T> setLookupTable(@NonNull WeightLookupTable<T> lookupTable) {
            this.lookupTable = lookupTable;
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

            vectors.iterator = this.iterator;
            vectors.lookupTable = this.lookupTable;

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
        private AtomicBoolean isRunning = new AtomicBoolean(false);
        private AtomicLong nextRandom;

        public AsyncSequencer(SequenceIterator<T> iterator) {
            this.iterator = iterator;
            this.buffer = new LinkedBlockingQueue<>();
//            this.linesCounter = linesCounter;
            this.setName("AsyncSequencer thread");
            this.nextRandom = new AtomicLong(workers + 1);
            this.iterator.reset();
        }

        @Override
        public void run() {
            isRunning.set(true);
            while (this.iterator.hasMoreSequences()) {

                // if buffered level is below limitLower, we're going to fetch limitUpper number of strings from fetcher
                if (buffer.size() < limitLower) {
                    AtomicInteger linesLoaded = new AtomicInteger(0);
                    while (linesLoaded.getAndIncrement() < limitUpper && this.iterator.hasMoreSequences() ) {
                        Sequence<T> document = this.iterator.nextSequence();

                        /*
                            We can't hope/assume that underlying iterator contains synchronized elements
                            That's why we're going to rebuild sequence from vocabulary
                          */
                        Sequence<T> newSequence = new Sequence<>();

                        for (T element: document.getElements()) {
                            newSequence.addElement(vocab.wordFor(element.getLabel()));
                        }
                        buffer.add(newSequence);

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
                    Sequence<T> sequence = digitizer.nextSentence();
                    /*
                            TODO: investigate, if fix needed here to become iteration-dependent, not line-position
                      */
                    double alpha = 0.025;

                    if (sequence == null || sequence.getElements().size() == 0) {
                        continue;
                    }

                    // getting back number of iterations
                    for (int i = 0; i < AbstractVectors.this.numIterations; i++) {
                        alpha = Math.max(AbstractVectors.this.minLearningRate, AbstractVectors.this.learningRate.get() * (1 - (1.0 * this.wordsCounter.get() / (double) this.totalWordsCount)));

                        AbstractVectors.this.trainSequence(sequence, nextRandom, alpha);

                        // increment processed word count, please note: this affects learningRate decay
                        this.wordsCounter.addAndGet(sequence.getElements().size());
                    }

                    // increment processed lines count
                    totalLines.incrementAndGet();
                    if (totalLines.get() % 100000 == 0) log.info("Epoch: " + this.epochNumber+ "; Words vectorized so far: " + this.wordsCounter.get() + ";  Lines vectorized so far: " + this.totalLines.get() + "; learningRate: " + alpha);
                } catch (Exception  e) {
                    e.printStackTrace();
                    throw new RuntimeException(e);
                }
            }
        }
    }
}
