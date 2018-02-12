package org.deeplearning4j.models.embeddings.learning.impl.elements;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
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
import org.nd4j.linalg.api.ops.aggregates.impl.AggregateCBOW;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.DeviceLocalNDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * CBOW implementation for DeepLearning4j
 *
 * @author raver119@gmail.com
 */
public class CBOW<T extends SequenceElement> implements ElementsLearningAlgorithm<T> {
    private VocabCache<T> vocabCache;
    private WeightLookupTable<T> lookupTable;
    private VectorsConfiguration configuration;

    private static final Logger logger = LoggerFactory.getLogger(CBOW.class);

    protected static double MAX_EXP = 6;

    protected int window;
    protected boolean useAdaGrad;
    protected double negative;
    protected double sampling;
    protected int[] variableWindows;

    @Getter
    @Setter
    protected DeviceLocalNDArray syn0, syn1, syn1Neg, expTable, table;

    protected ThreadLocal<List<Aggregate>> batches = new ThreadLocal<>();

    public List<Aggregate> getBatch() {
        return batches.get();
    }

    @Override
    public String getCodeName() {
        return "CBOW";
    }

    @Override
    public void configure(@NonNull VocabCache<T> vocabCache, @NonNull WeightLookupTable<T> lookupTable,
                    @NonNull VectorsConfiguration configuration) {
        this.vocabCache = vocabCache;
        this.lookupTable = lookupTable;
        this.configuration = configuration;

        this.window = configuration.getWindow();
        this.useAdaGrad = configuration.isUseAdaGrad();
        this.negative = configuration.getNegative();
        this.sampling = configuration.getSampling();

        if (configuration.getNegative() > 0) {
            if (((InMemoryLookupTable<T>) lookupTable).getSyn1Neg() == null) {
                logger.info("Initializing syn1Neg...");
                ((InMemoryLookupTable<T>) lookupTable).setUseHS(configuration.isUseHierarchicSoftmax());
                ((InMemoryLookupTable<T>) lookupTable).setNegative(configuration.getNegative());
                ((InMemoryLookupTable<T>) lookupTable).resetWeights(false);
            }
        }


        this.syn0 = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn0());
        this.syn1 = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1());
        this.syn1Neg = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1Neg());
        this.expTable = new DeviceLocalNDArray(Nd4j.create(((InMemoryLookupTable<T>) lookupTable).getExpTable()));
        this.table = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getTable());
        this.variableWindows = configuration.getVariableWindows();
    }

    /**
     * CBOW doesn't involve any pretraining
     *
     * @param iterator
     */
    @Override
    public void pretrain(SequenceIterator<T> iterator) {
        // no-op
    }

    @Override
    public void finish() {
        if (batches != null && batches.get() != null && !batches.get().isEmpty()) {
            Nd4j.getExecutioner().exec(batches.get());
            batches.get().clear();
        }
    }

    @Override
    public double learnSequence(Sequence<T> sequence, AtomicLong nextRandom, double learningRate) {
        Sequence<T> tempSequence = sequence;
        if (sampling > 0)
            tempSequence = applySubsampling(sequence, nextRandom);

        int currentWindow = window;

        if (variableWindows != null && variableWindows.length != 0) {
            currentWindow = variableWindows[RandomUtils.nextInt(variableWindows.length)];
        }

        for (int i = 0; i < tempSequence.getElements().size(); i++) {
            nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
            cbow(i, tempSequence.getElements(), (int) nextRandom.get() % currentWindow, nextRandom, learningRate,
                            currentWindow);
        }

        return 0;
    }

    @Override
    public boolean isEarlyTerminationHit() {
        return false;
    }

    public void iterateSample(T currentWord, int[] windowWords, AtomicLong nextRandom, double alpha,
                    boolean isInference, int numLabels, boolean trainWords, INDArray inferenceVector) {
        int[] idxSyn1 = null;
        int[] codes = null;

        if (configuration.isUseHierarchicSoftmax()) {
            idxSyn1 = new int[currentWord.getCodeLength()];
            codes = new int[currentWord.getCodeLength()];
            for (int p = 0; p < currentWord.getCodeLength(); p++) {
                if (currentWord.getPoints().get(p) < 0)
                    continue;

                codes[p] = currentWord.getCodes().get(p);
                idxSyn1[p] = currentWord.getPoints().get(p);
            }
        } else {
            idxSyn1 = new int[0];
            codes = new int[0];
        }


        if (negative > 0) {
            if (syn1Neg == null) {
                ((InMemoryLookupTable<T>) lookupTable).initNegative();
                syn1Neg = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1Neg());
            }
        }

        if (batches.get() == null)
            batches.set(new ArrayList<Aggregate>());

        AggregateCBOW cbow = new AggregateCBOW(syn0.get(), syn1.get(), syn1Neg.get(), expTable.get(), table.get(),
                        currentWord.getIndex(), windowWords, idxSyn1, codes, (int) negative, currentWord.getIndex(),
                        lookupTable.layerSize(), alpha, nextRandom.get(), vocabCache.numWords(), numLabels, trainWords,
                        inferenceVector);
        nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));

        if (!isInference) {
            batches.get().add(cbow);
            if (batches.get().size() > 4096) {
                Nd4j.getExecutioner().exec(batches.get());
                batches.get().clear();
            }
        } else
            Nd4j.getExecutioner().exec(cbow);

    }

    public void cbow(int i, List<T> sentence, int b, AtomicLong nextRandom, double alpha, int currentWindow) {
        int end = window * 2 + 1 - b;

        T currentWord = sentence.get(i);

        List<Integer> intsList = new ArrayList<>();
        for (int a = b; a < end; a++) {
            if (a != currentWindow) {
                int c = i - currentWindow + a;
                if (c >= 0 && c < sentence.size()) {
                    T lastWord = sentence.get(c);

                    intsList.add(lastWord.getIndex());
                }
            }
        }

        int[] windowWords = new int[intsList.size()];
        for (int x = 0; x < windowWords.length; x++) {
            windowWords[x] = intsList.get(x);
        }

        // we don't allow inference from main loop here
        iterateSample(currentWord, windowWords, nextRandom, alpha, false, 0, true, null);

        if (batches != null && batches.get() != null && batches.get().size() >= configuration.getBatchSize()) {
            Nd4j.getExecutioner().exec(batches.get());
            batches.get().clear();
        }
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
}
