/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.models.embeddings.learning.impl.elements;

import lombok.NonNull;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.glove.AbstractCoOccurrences;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.legacy.AdaGrad;
import org.nd4j.linalg.primitives.Counter;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * GloVe LearningAlgorithm implementation for SequenceVectors
 *
 *
 * @author raver119@gmail.com
 */
public class GloVe<T extends SequenceElement> implements ElementsLearningAlgorithm<T> {

    private VocabCache<T> vocabCache;
    private AbstractCoOccurrences<T> coOccurrences;
    private WeightLookupTable<T> lookupTable;
    private VectorsConfiguration configuration;

    private AtomicBoolean isTerminate = new AtomicBoolean(false);

    private INDArray syn0;

    private double xMax;
    private boolean shuffle;
    private boolean symmetric;
    protected double alpha = 0.75d;
    protected double learningRate = 0.0d;
    protected int maxmemory = 0;
    protected int batchSize = 1000;

    private AdaGrad weightAdaGrad;
    private AdaGrad biasAdaGrad;
    private INDArray bias;

    private int workers = Runtime.getRuntime().availableProcessors();

    private int vectorLength;

    private static final Logger log = LoggerFactory.getLogger(GloVe.class);

    @Override
    public String getCodeName() {
        return "GloVe";
    }

    @Override
    public void finish() {
        log.info("GloVe finalizer...");
    }

    @Override
    public void configure(@NonNull VocabCache<T> vocabCache, @NonNull WeightLookupTable<T> lookupTable,
                    @NonNull VectorsConfiguration configuration) {
        this.vocabCache = vocabCache;
        this.lookupTable = lookupTable;
        this.configuration = configuration;

        this.syn0 = ((InMemoryLookupTable<T>) lookupTable).getSyn0();


        this.vectorLength = configuration.getLayersSize();

        if (this.learningRate == 0.0d)
            this.learningRate = configuration.getLearningRate();



        weightAdaGrad = new AdaGrad(new int[] {this.vocabCache.numWords() + 1, vectorLength}, learningRate);
        bias = Nd4j.create(syn0.rows());

        // FIXME: int cast
        biasAdaGrad = new AdaGrad(ArrayUtil.toInts(bias.shape()), this.learningRate);

        //  maxmemory = Runtime.getRuntime().maxMemory() - (vocabCache.numWords() * vectorLength * 2 * 8);

        log.info("GloVe params: {Max Memory: [" + maxmemory + "], Learning rate: [" + this.learningRate + "], Alpha: ["
                        + alpha + "], xMax: [" + xMax + "], Symmetric: [" + symmetric + "], Shuffle: [" + shuffle
                        + "]}");
    }

    /**
     * pretrain is used to build CoOccurrence matrix for GloVe algorithm
     * @param iterator
     */
    @Override
    public void pretrain(@NonNull SequenceIterator<T> iterator) {
        // CoOccurence table should be built here
        coOccurrences = new AbstractCoOccurrences.Builder<T>()
                        // TODO: symmetric should be handled via VectorsConfiguration
                        .symmetric(this.symmetric).windowSize(configuration.getWindow()).iterate(iterator)
                        .workers(workers).vocabCache(vocabCache).maxMemory(maxmemory).build();

        coOccurrences.fit();
    }

    /**
     * Learns sequence using GloVe algorithm
     *
     * @param sequence
     * @param nextRandom
     * @param learningRate
     */
    @Override
    public synchronized double learnSequence(@NonNull Sequence<T> sequence, @NonNull AtomicLong nextRandom,
                    double learningRate) {
        /*
                GloVe learning algorithm is implemented like a hack over settled ElementsLearningAlgorithm mechanics. It's called in SequenceVectors context, but actually only for the first call.
                All subsequent calls will met early termination condition, and will be successfully ignored. But since elements vectors will be updated within first call,
                this will allow compatibility with everything beyond this implementaton
         */
        if (isTerminate.get())
            return 0;

        final AtomicLong pairsCount = new AtomicLong(0);
        final Counter<Integer> errorCounter = new Counter<>();

        //List<Pair<T, T>> coList = coOccurrences.coOccurrenceList();

        for (int i = 0; i < configuration.getEpochs(); i++) {

            // TODO: shuffle should be built in another way.
            //if (shuffle)
            //Collections.shuffle(coList);

            Iterator<Pair<Pair<T, T>, Double>> pairs = coOccurrences.iterator();

            List<GloveCalculationsThread> threads = new ArrayList<>();
            for (int x = 0; x < workers; x++) {
                threads.add(x, new GloveCalculationsThread(i, x, pairs, pairsCount, errorCounter));
                threads.get(x).start();
            }



            for (int x = 0; x < workers; x++) {
                try {
                    threads.get(x).join();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }

            log.info("Processed [" + pairsCount.get() + "] pairs, Error was [" + errorCounter.getCount(i) + "]");
        }

        isTerminate.set(true);
        return 0;
    }

    /**
     *  Since GloVe is learning representations using elements CoOccurences, all training is done in GloVe class internally, so only first thread will execute learning process,
     *  and the rest of parent threads will just exit learning process
     *
     * @return True, if training should stop, False otherwise.
     */
    @Override
    public synchronized boolean isEarlyTerminationHit() {
        return isTerminate.get();
    }

    private double iterateSample(T element1, T element2, double score) {
        //prediction: input + bias
        if (element1.getIndex() < 0 || element1.getIndex() >= syn0.rows())
            throw new IllegalArgumentException("Illegal index for word " + element1.getLabel());
        if (element2.getIndex() < 0 || element2.getIndex() >= syn0.rows())
            throw new IllegalArgumentException("Illegal index for word " + element2.getLabel());

        INDArray w1Vector = syn0.slice(element1.getIndex());
        INDArray w2Vector = syn0.slice(element2.getIndex());


        //w1 * w2 + bias
        double prediction = Nd4j.getBlasWrapper().dot(w1Vector, w2Vector);
        prediction += bias.getDouble(element1.getIndex()) + bias.getDouble(element2.getIndex()) - Math.log(score);

        double fDiff = (score > xMax) ? prediction : Math.pow(score / xMax, alpha) * prediction; // Math.pow(Math.min(1.0,(score / maxCount)),xMax);

        //        double fDiff = score > xMax ? prediction :  weight * (prediction - Math.log(score));

        if (Double.isNaN(fDiff))
            fDiff = Nd4j.EPS_THRESHOLD;
        //amount of change
        double gradient = fDiff * learningRate;

        //note the update step here: the gradient is
        //the gradient of the OPPOSITE word
        //for adagrad we will use the index of the word passed in
        //for the gradient calculation we will use the context vector
        update(element1, w1Vector, w2Vector, gradient);
        update(element2, w2Vector, w1Vector, gradient);
        return 0.5 * fDiff * prediction;
    }

    private void update(T element1, INDArray wordVector, INDArray contextVector, double gradient) {
        //gradient for word vectors
        INDArray grad1 = contextVector.mul(gradient);
        // FIXME: int cast
        INDArray update = weightAdaGrad.getGradient(grad1, element1.getIndex(), ArrayUtil.toInts(syn0.shape()));

        //update vector
        wordVector.subi(update);

        double w1Bias = bias.getDouble(element1.getIndex());
        // FIXME: int cast
        double biasGradient = biasAdaGrad.getGradient(gradient, element1.getIndex(), ArrayUtil.toInts(bias.shape()));
        double update2 = w1Bias - biasGradient;
        bias.putScalar(element1.getIndex(), update2);
    }

    private class GloveCalculationsThread extends Thread implements Runnable {
        private final int threadId;
        private final int epochId;
        //        private final AbstractCoOccurrences<T> coOccurrences;
        private final Iterator<Pair<Pair<T, T>, Double>> coList;

        private final AtomicLong pairsCounter;
        private final Counter<Integer> errorCounter;

        public GloveCalculationsThread(int epochId, int threadId, @NonNull Iterator<Pair<Pair<T, T>, Double>> pairs,
                        @NonNull AtomicLong pairsCounter, @NonNull Counter<Integer> errorCounter) {
            this.epochId = epochId;
            this.threadId = threadId;
            //  this.coOccurrences = coOccurrences;

            this.pairsCounter = pairsCounter;
            this.errorCounter = errorCounter;

            coList = pairs;

            this.setName("GloVe ELA t." + this.threadId);
        }

        @Override
        public void run() {
            //            int startPosition = threadId * (coList.size() / workers);
            //            int stopPosition = (threadId + 1) *  (coList.size() / workers);
            //            log.info("Total size: [" + coList.size() + "], thread start: [" + startPosition + "], thread stop: [" + stopPosition + "]");
            while (coList.hasNext()) {

                // now we fetch pairs into batch
                List<Pair<Pair<T, T>, Double>> pairs = new ArrayList<>();
                int cnt = 0;
                while (coList.hasNext() && cnt < batchSize) {
                    pairs.add(coList.next());
                    cnt++;
                }

                if (shuffle)
                    Collections.shuffle(pairs);

                Iterator<Pair<Pair<T, T>, Double>> iterator = pairs.iterator();

                while (iterator.hasNext()) {
                    // now for each pair do appropriate training
                    Pair<Pair<T, T>, Double> pairDoublePair = iterator.next();

                    // That's probably ugly and probably should be improved somehow

                    T element1 = pairDoublePair.getFirst().getFirst();
                    T element2 = pairDoublePair.getFirst().getSecond();
                    double weight = pairDoublePair.getSecond(); //coOccurrences.getCoOccurrenceCount(element1, element2);
                    if (weight <= 0) {
                        //                    log.warn("Skipping pair ("+ element1.getLabel()+", " + element2.getLabel()+")");
                        pairsCounter.incrementAndGet();
                        continue;
                    }

                    errorCounter.incrementCount(epochId, iterateSample(element1, element2, weight));
                    if (pairsCounter.incrementAndGet() % 1000000 == 0) {
                        log.info("Processed [" + pairsCounter.get() + "] word pairs so far...");
                    }
                }

            }
        }
    }

    public static class Builder<T extends SequenceElement> {

        protected double xMax = 100.0d;
        protected double alpha = 0.75d;
        protected double learningRate = 0.0d;

        protected boolean shuffle = false;
        protected boolean symmetric = false;
        protected int maxmemory = 0;

        protected int batchSize = 1000;

        public Builder() {

        }

        /**
         * This parameter specifies batch size for each thread. Also, if shuffle == TRUE, this batch will be shuffled before processing. Default value: 1000;
         *
         * @param batchSize
         * @return
         */
        public Builder<T> batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }


        /**
         * Initial learning rate; default 0.05
         *
         * @param eta
         * @return
         */
        public Builder<T> learningRate(double eta) {
            this.learningRate = eta;
            return this;
        }

        /**
         * Parameter in exponent of weighting function; default 0.75
         *
         * @param alpha
         * @return
         */
        public Builder<T> alpha(double alpha) {
            this.alpha = alpha;
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
        public Builder<T> maxMemory(int gbytes) {
            this.maxmemory = gbytes;
            return this;
        }

        /**
         * Parameter specifying cutoff in weighting function; default 100.0
         *
         * @param xMax
         * @return
         */
        public Builder<T> xMax(double xMax) {
            this.xMax = xMax;
            return this;
        }

        /**
         * Parameter specifying, if cooccurrences list should be shuffled between training epochs
         *
         * @param reallyShuffle
         * @return
         */
        public Builder<T> shuffle(boolean reallyShuffle) {
            this.shuffle = reallyShuffle;
            return this;
        }

        /**
         * Parameters specifying, if cooccurrences list should be build into both directions from any current word.
         *
         * @param reallySymmetric
         * @return
         */
        public Builder<T> symmetric(boolean reallySymmetric) {
            this.symmetric = reallySymmetric;
            return this;
        }

        public GloVe<T> build() {
            GloVe<T> ret = new GloVe<>();
            ret.symmetric = this.symmetric;
            ret.shuffle = this.shuffle;
            ret.xMax = this.xMax;
            ret.alpha = this.alpha;
            ret.learningRate = this.learningRate;
            ret.maxmemory = this.maxmemory;
            ret.batchSize = this.batchSize;

            return ret;
        }
    }
}
