package org.deeplearning4j.models.embeddings.learning.impl.elements;

import com.google.common.util.concurrent.AtomicDouble;
import lombok.NonNull;
import org.apache.commons.lang.ArrayUtils;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.glove.AbstractCoOccurrences;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdaGrad;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * GloVe implementation for SequenceVectors
 *
 * @author raver119@gmail.com
 */
public  class GloVe<T extends SequenceElement> implements ElementsLearningAlgorithm<T> {

    private VocabCache<T> vocabCache;
    private AbstractCoOccurrences<T> coOccurrences;
    private WeightLookupTable<T> lookupTable;
    private VectorsConfiguration configuration;

    private AtomicBoolean isTerminate = new AtomicBoolean(false);

    private INDArray syn0;

    private double xMax;
    private boolean shuffle;
    private boolean symmetric;
    private int maxCount;

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
    public void configure(@NonNull VocabCache<T> vocabCache, @NonNull WeightLookupTable<T> lookupTable, @NonNull VectorsConfiguration configuration) {
        this.vocabCache = vocabCache;
        this.lookupTable = lookupTable;
        this.configuration = configuration;

        this.syn0 = ((InMemoryLookupTable<T>)lookupTable).getSyn0();

        this.xMax = configuration.getXMax();
        this.symmetric = configuration.isSymmetric();
        this.shuffle = configuration.isShuffle();
        this.maxCount = configuration.getMaxCount();
        this.vectorLength = configuration.getLayersSize();


        log.info("Vectors configuration: " + configuration.toJson());

        weightAdaGrad = new AdaGrad(new int[]{this.vocabCache.numWords() + 1, vectorLength}, this.configuration.getLearningRate());
        bias = Nd4j.create(syn0.rows());
        biasAdaGrad = new AdaGrad(bias.shape(), this.configuration.getLearningRate());
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
                .symmetric(this.symmetric)
                .windowSize(configuration.getWindow())
                .iterate(iterator)
                .workers(workers)
                .vocabCache(vocabCache)
                .build();

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
    public synchronized void learnSequence(@NonNull Sequence<T> sequence, @NonNull AtomicLong nextRandom, double learningRate) {
        /*
                GloVe learning algorithm is implemented like a hack, over existing code base. It's called in SequenceVectors context, but actually only for the first call.
                All subsequent calls will met early termination condition, and will be successfully ignored. But since elements vectors will be updated within first call,
                this will allow compatibility with everything beyond this implementaton
         */
        if (isTerminate.get()) return;

        final AtomicInteger pairsCount = new AtomicInteger(0);
        final Counter<Integer> errorCounter = new Counter<>();

        List<Pair<T, T>> coList = coOccurrences.coOccurrenceList();

        for (int i = 0; i < configuration.getEpochs(); i++ ) {

            if (shuffle)
                Collections.shuffle(coList);


            List<GloveCalculationsThread> threads = new ArrayList<>();
            for (int x = 0; x < workers; x++) {
                threads.add(x, new GloveCalculationsThread(i, x, coOccurrences, coList, pairsCount, errorCounter));
                threads.get(x).start();
            }



            for (int x = 0; x < workers; x++) {
                try {
                    threads.get(x).join();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }

            log.info("Processed ["+ pairsCount.get()+"] pairs, out of [" + ( coList.size() * configuration.getEpochs())+"]; Error was ["+ errorCounter.getCount(i) +"]");
        }

        isTerminate.set(true);
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
        INDArray w1Vector = syn0.slice(element1.getIndex());
        INDArray w2Vector = syn0.slice(element2.getIndex());
        //prediction: input + bias
        if(element1.getIndex() < 0 || element1.getIndex() >= syn0.rows())
            throw new IllegalArgumentException("Illegal index for word " + element1.getLabel());
        if(element2.getIndex() < 0 || element2.getIndex() >= syn0.rows())
            throw new IllegalArgumentException("Illegal index for word " + element2.getLabel());


        //w1 * w2 + bias
        double prediction = Nd4j.getBlasWrapper().dot(w1Vector,w2Vector);
        prediction +=  bias.getDouble(element1.getIndex()) + bias.getDouble(element2.getIndex());

        double weight = Math.pow(Math.min(1.0,(score / maxCount)),xMax);

        double fDiff = score > xMax ? prediction :  weight * (prediction - Math.log(score));
        if(Double.isNaN(fDiff))
            fDiff = Nd4j.EPS_THRESHOLD;
        //amount of change
        double gradient =  fDiff;

        //note the update step here: the gradient is
        //the gradient of the OPPOSITE word
        //for adagrad we will use the index of the word passed in
        //for the gradient calculation we will use the context vector
        update(element1, w1Vector, w2Vector, gradient);
        update(element2, w2Vector, w1Vector, gradient);
        return fDiff;
    }

    private void update(T element1, INDArray wordVector, INDArray contextVector, double gradient) {
        //gradient for word vectors
        INDArray grad1 =  contextVector.mul(gradient);
        INDArray update = weightAdaGrad.getGradient(grad1,element1.getIndex(),syn0.shape());

        //update vector
        wordVector.subi(update);

        double w1Bias = bias.getDouble(element1.getIndex());
        double biasGradient = biasAdaGrad.getGradient(gradient,element1.getIndex(),bias.shape());
        double update2 = w1Bias - biasGradient;
        bias.putScalar(element1.getIndex(),update2);
    }

    private class GloveCalculationsThread extends Thread implements Runnable {
        private final int threadId;
        private final int epochId;
        private final AbstractCoOccurrences<T> coOccurrences;
        private final List<Pair<T, T>> coList;

        private final AtomicInteger pairsCounter;
        private final Counter<Integer> errorCounter;

        public GloveCalculationsThread(int epochId, int threadId, @NonNull AbstractCoOccurrences<T> coOccurrences, @NonNull List<Pair<T, T>> pairs, @NonNull AtomicInteger pairsCounter, @NonNull Counter<Integer> errorCounter) {
            this.epochId = epochId;
            this.threadId = threadId;
            this.coOccurrences = coOccurrences;

            this.pairsCounter = pairsCounter;
            this.errorCounter = errorCounter;

            coList = pairs;

            this.setName("GloVe ElementsLearningAlgorithm thread " + this.threadId);
        }

        @Override
        public void run() {
            int startPosition = threadId * (coList.size() / workers);
            int stopPosition = (threadId + 1) *  (coList.size() / workers);
//            log.info("Total size: [" + coList.size() + "], thread start: [" + startPosition + "], thread stop: [" + stopPosition + "]");
            for (int x = startPosition; x < stopPosition; x++) {
                // no for each pair do appropriate training
                T element1 = coList.get(x).getFirst();
                T element2 = coList.get(x).getSecond();
                double weight = coOccurrences.getCoOccurrenceCount(element1, element2);
                if (weight <= 0) {
//                    log.warn("Skipping pair ("+ element1.getLabel()+", " + element2.getLabel()+")");
                    pairsCounter.incrementAndGet();
                    continue;
                }

                errorCounter.incrementCount(epochId, iterateSample(element1, element2, weight));
                if (pairsCounter.getAndIncrement() % 10000 == 0) {
           //         log.info("Processed [" + pairsCounter.get() + "] word pairs so far...");
                }
            }
        }
    }
}
