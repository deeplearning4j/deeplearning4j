package org.deeplearning4j.models.embeddings.learning.impl.elements;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.api.ops.aggregates.impl.AggregateSkipGram;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.DeviceLocalNDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Skip-Gram implementation for dl4j SequenceVectors
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class SkipGram<T extends SequenceElement> implements ElementsLearningAlgorithm<T> {
    protected VocabCache<T> vocabCache;
    protected WeightLookupTable<T> lookupTable;
    protected VectorsConfiguration configuration;

    protected int window;
    protected boolean useAdaGrad;
    protected double negative;
    protected double sampling;
    protected int[] variableWindows;
    protected int vectorLength;

    @Getter
    @Setter
    protected DeviceLocalNDArray syn0, syn1, syn1Neg, table, expTable;

    protected ThreadLocal<List<Aggregate>> batches = new ThreadLocal<>();

    /**
     * Dummy construction is required for reflection
     */
    public SkipGram() {

    }

    public List<Aggregate> getBatch() {
        return batches.get();
    }

    /**
     * Returns implementation code name
     *
     * @return
     */
    @Override
    public String getCodeName() {
        return "SkipGram";
    }

    /**
     * SkipGram initialization over given vocabulary and WeightLookupTable
     *
     * @param vocabCache
     * @param lookupTable
     * @param configuration
     */
    @Override
    public void configure(@NonNull VocabCache<T> vocabCache, @NonNull WeightLookupTable<T> lookupTable,
                    @NonNull VectorsConfiguration configuration) {
        this.vocabCache = vocabCache;
        this.lookupTable = lookupTable;
        this.configuration = configuration;

        if (configuration.getNegative() > 0) {
            if (((InMemoryLookupTable<T>) lookupTable).getSyn1Neg() == null) {
                log.info("Initializing syn1Neg...");
                ((InMemoryLookupTable<T>) lookupTable).setUseHS(configuration.isUseHierarchicSoftmax());
                ((InMemoryLookupTable<T>) lookupTable).setNegative(configuration.getNegative());
                ((InMemoryLookupTable<T>) lookupTable).resetWeights(false);
            }
        }

        this.expTable = new DeviceLocalNDArray(Nd4j.create(((InMemoryLookupTable<T>) lookupTable).getExpTable()));
        this.syn0 = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn0());
        this.syn1 = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1());
        this.syn1Neg = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1Neg());
        this.table = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getTable());



        this.window = configuration.getWindow();
        this.useAdaGrad = configuration.isUseAdaGrad();
        this.negative = configuration.getNegative();
        this.sampling = configuration.getSampling();
        this.variableWindows = configuration.getVariableWindows();

        this.vectorLength = configuration.getLayersSize();
    }

    /**
     * SkipGram doesn't involves any pretraining
     *
     * @param iterator
     */
    @Override
    public void pretrain(SequenceIterator<T> iterator) {
        // no-op
    }

    public Sequence<T> applySubsampling(@NonNull Sequence<T> sequence, @NonNull AtomicLong nextRandom) {
        Sequence<T> result = new Sequence<>();

        // subsampling implementation, if subsampling threshold met, just continue to next element
        if (sampling > 0) {
            result.setSequenceId(sequence.getSequenceId());
            if (sequence.getSequenceLabels() != null)
                result.setSequenceLabels(sequence.getSequenceLabels());
            if (sequence.getSequenceLabel() != null)
                result.setSequenceLabel(sequence.getSequenceLabel());

            for (T element : sequence.getElements()) {
                double numWords = vocabCache.totalWordOccurrences();
                double ran = (Math.sqrt(element.getElementFrequency() / (sampling * numWords)) + 1)
                                * (sampling * numWords) / element.getElementFrequency();

                nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));

                if (ran < (nextRandom.get() & 0xFFFF) / (double) 65536) {
                    continue;
                }
                result.addElement(element);
            }
            return result;
        } else
            return sequence;
    }

    /**
     * Learns sequence using SkipGram algorithm
     *
     * @param sequence
     * @param nextRandom
     * @param learningRate
     */
    @Override
    public double learnSequence(@NonNull Sequence<T> sequence, @NonNull AtomicLong nextRandom, double learningRate) {
        Sequence<T> tempSequence = sequence;
        if (sampling > 0)
            tempSequence = applySubsampling(sequence, nextRandom);

        double score = 0.0;

        int currentWindow = window;

        if (variableWindows != null && variableWindows.length != 0) {
            currentWindow = variableWindows[RandomUtils.nextInt(variableWindows.length)];
        }

        for (int i = 0; i < tempSequence.getElements().size(); i++) {
            nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
            score = skipGram(i, tempSequence.getElements(), (int) nextRandom.get() % currentWindow, nextRandom,
                            learningRate, currentWindow);
        }

        if (batches != null && batches.get() != null && batches.get().size() >= configuration.getBatchSize()) {
            Nd4j.getExecutioner().exec(batches.get());
            batches.get().clear();
        }

        return score;
    }

    @Override
    public void finish() {
        if (batches != null && batches.get() != null && !batches.get().isEmpty()) {
            Nd4j.getExecutioner().exec(batches.get());
            batches.get().clear();
        }
    }

    /**
     * SkipGram has no reasons for early termination ever.
     *
     * @return
     */
    @Override
    public boolean isEarlyTerminationHit() {
        return false;
    }

    private double skipGram(int i, List<T> sentence, int b, AtomicLong nextRandom, double alpha, int currentWindow) {
        final T word = sentence.get(i);
        if (word == null || sentence.isEmpty())
            return 0.0;

        double score = 0.0;
        int cnt = 0;

        int end = currentWindow * 2 + 1 - b;
        for (int a = b; a < end; a++) {
            if (a != currentWindow) {
                int c = i - currentWindow + a;
                if (c >= 0 && c < sentence.size()) {
                    T lastWord = sentence.get(c);
                    score = iterateSample(word, lastWord, nextRandom, alpha, false, null);

                }
            }
        }

        return score;
    }

    public double iterateSample(T w1, T lastWord, AtomicLong nextRandom, double alpha, boolean isInference,
                    INDArray inferenceVector) {
        if (w1 == null || lastWord == null || (lastWord.getIndex() < 0 && !isInference)
                        || w1.getIndex() == lastWord.getIndex() || w1.getLabel().equals("STOP")
                        || lastWord.getLabel().equals("STOP") || w1.getLabel().equals("UNK")
                        || lastWord.getLabel().equals("UNK")) {
            return 0.0;
        }


        double score = 0.0;

        int[] idxSyn1 = null;
        int[] codes = null;
        if (configuration.isUseHierarchicSoftmax()) {
            idxSyn1 = new int[w1.getCodeLength()];
            codes = new int[w1.getCodeLength()];
            for (int i = 0; i < w1.getCodeLength(); i++) {
                int code = w1.getCodes().get(i);
                int point = w1.getPoints().get(i);
                if (point >= vocabCache.numWords() || point < 0)
                    continue;

                codes[i] = code;
                idxSyn1[i] = point;
            }
        } else {
            idxSyn1 = new int[0];
            codes = new int[0];
        }


        int target = w1.getIndex();
        //negative sampling
        if (negative > 0) {
            if (syn1Neg == null) {
                ((InMemoryLookupTable<T>) lookupTable).initNegative();
                syn1Neg = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1Neg());
            }
        }

        if (batches.get() == null) {
            batches.set(new ArrayList<Aggregate>());
        }

        //log.info("VocabWords: {}; lastWordIndex: {}; syn1neg: {}", vocabCache.numWords(), lastWord.getIndex(), syn1Neg.get().rows());

        AggregateSkipGram sg = new AggregateSkipGram(syn0.get(), syn1.get(), syn1Neg.get(), expTable.get(), table.get(),
                        lastWord.getIndex(), idxSyn1, codes, (int) negative, target, vectorLength, alpha,
                        nextRandom.get(), vocabCache.numWords(), inferenceVector);
        nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));

        if (!isInference) {
            batches.get().add(sg);
            if (batches.get().size() > 0) {
                Nd4j.getExecutioner().exec(batches.get());
                batches.get().clear();
            }
        } else
            Nd4j.getExecutioner().exec(sg);

        return score;
    }
}
