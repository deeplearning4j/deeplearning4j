/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.models.embeddings.learning.impl.elements;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.RandomUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.api.ops.impl.nlp.CbowInference;
import org.nd4j.linalg.api.ops.impl.nlp.CbowRound;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.DeviceLocalNDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

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
    protected int workers = Runtime.getRuntime().availableProcessors();

    public int getWorkers() {
        return workers;
    }

    public void setWorkers(int workers) {
        this.workers = workers;
    }

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
                lookupTable.resetWeights(false);
            }
        }


        this.syn0 = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn0());
        this.syn1 = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1());
        this.syn1Neg = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1Neg());
        this.expTable = new DeviceLocalNDArray(Nd4j.create(((InMemoryLookupTable<T>) lookupTable).getExpTable(),
                new long[]{((InMemoryLookupTable<T>) lookupTable).getExpTable().length}, syn0.get() == null ? DataType.DOUBLE :  syn0.get().dataType()));
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
    public double learnSequence(Sequence<T> sequence, AtomicLong nextRandom, double learningRate,
                                BatchSequences<T> batchSequences) {
        Sequence<T> tempSequence = sequence;

        if (sampling > 0)
            tempSequence = applySubsampling(sequence, nextRandom);

        int currentWindow = window;

        if (variableWindows != null && variableWindows.length != 0) {
            currentWindow = variableWindows[RandomUtils.nextInt(0, variableWindows.length)];
        }

        for (int i = 0; i < tempSequence.getElements().size(); i++) {
            nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
            cbow(i, tempSequence.getElements(), (int) nextRandom.get() % currentWindow, nextRandom, learningRate,
                    currentWindow, batchSequences);
        }

        return 0;
    }

    @Override
    public double learnSequence(Sequence<T> sequence, AtomicLong nextRandom, double learningRate) {
        Sequence<T> tempSequence = sequence;
        if (sampling > 0)
            tempSequence = applySubsampling(sequence, nextRandom);

        int currentWindow = window;

        if (variableWindows != null && variableWindows.length != 0) {
            currentWindow = variableWindows[RandomUtils.nextInt(0, variableWindows.length)];
        }

        for (int i = 0; i < tempSequence.getElements().size(); i++) {
            nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
            cbow(i, tempSequence.getElements(), (int) nextRandom.get() % currentWindow, nextRandom, learningRate,
                    currentWindow, null);
        }

        return 0;
    }

    @Override
    public boolean isEarlyTerminationHit() {
        return false;
    }


    public CustomOp iterateSampleOp(T currentWord, int[] windowWords, boolean[] wordStatuses, AtomicLong nextRandom, double alpha,
                                    boolean isInference, int numLabels, boolean trainWords, INDArray inferenceVector) {
        int[] idxSyn1 = null;
        byte[] codes = null;


        if (configuration.isUseHierarchicSoftmax()) {
            idxSyn1 = new int[currentWord.getCodeLength()];
            codes = new byte[currentWord.getCodeLength()];
            for (int p = 0; p < currentWord.getCodeLength(); p++) {
                if (currentWord.getPoints().get(p) < 0)
                    continue;

                codes[p] = currentWord.getCodes().get(p);
                idxSyn1[p] = currentWord.getPoints().get(p);
            }
        } else {
            idxSyn1 = new int[0];
            codes = new byte[0];
        }


        if (negative > 0) {
            if (syn1Neg == null) {
                ((InMemoryLookupTable<T>) lookupTable).initNegative();
                syn1Neg = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1Neg());
            }
        }

        if (batches.get() == null)
            batches.set(new ArrayList<>());



        boolean useHS = configuration.isUseHierarchicSoftmax();
        boolean useNegative = configuration.getNegative() > 0;

        int[] inputStatuses = new int[windowWords.length];
        for (int i = 0; i < windowWords.length; ++i) {
            if (i < wordStatuses.length)
                inputStatuses[i] = wordStatuses[i] ? 1 : 0;
            else
                inputStatuses[i] = -1;
        }



        CbowInference cbowInference = null;
        if (useHS && useNegative) {
            cbowInference = CbowInference.builder()
                    .ngStarter(currentWord.getIndex())
                    .target(currentWord.getIndex())
                    .randomValue((int) nextRandom.get())
                    .nsRounds((int) negative)
                    .syn0(syn0.get())
                    .syn1(syn1.get())
                    .syn1Neg(syn1Neg.get())
                    .expTable(expTable.get())
                    .negTable(table.get() == null ? Nd4j.empty(syn0.get().dataType()) : table.get())
                    .codes(codes)
                    .indices(idxSyn1)
                    .preciseMode(configuration.isPreciseMode())
                    .alpha(alpha)
                    .inferenceVector( inferenceVector != null ? inferenceVector : Nd4j.empty(syn0.get().dataType()))
                    .numWorkers(workers)
                    .build();

        }
        else if (useHS) {
            cbowInference = CbowInference.builder()
                    .target(currentWord.getIndex())
                    .ngStarter(currentWord.getIndex())
                    .randomValue((int) nextRandom.get())
                    .context(windowWords)
                    .expTable(expTable.get())
                    .lockedWords(inputStatuses)
                    .syn0(syn0.get())
                    .syn1(syn1.get())
                    .syn1Neg(Nd4j.empty(syn0.get().dataType()))
                    .negTable(table.get() == null ? Nd4j.empty(syn0.get().dataType()) : table.get())
                    .codes(codes)
                    .indices(new int[0])
                    .numLabels(numLabels)
                    .alpha(alpha)
                    .preciseMode(configuration.isPreciseMode())
                    .inferenceVector(inferenceVector != null ? inferenceVector : Nd4j.empty(syn0.get().dataType()))
                    .numWorkers(workers)
                    .build();

        }
        else if (useNegative) {
            cbowInference = CbowInference.builder()
                    .target(currentWord.getIndex())
                    .ngStarter(currentWord.getIndex())
                    .randomValue((int) nextRandom.get())
                    .context(windowWords)
                    .expTable(expTable.get())
                    .negTable(table.get())
                    .lockedWords(inputStatuses)
                    .syn0(syn0.get())
                    .syn1(Nd4j.empty(syn0.get().dataType()))
                    .syn1Neg(Nd4j.empty(syn0.get().dataType()))
                    .codes(new byte[0])
                    .indices(new int[0])
                    .numLabels(numLabels)
                    .alpha(alpha)
                    .preciseMode(configuration.isPreciseMode())
                    .inferenceVector(inferenceVector != null ? inferenceVector : Nd4j.empty(syn0.get().dataType()))
                    .numWorkers(workers)
                    .build();
        }

        nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
        return cbowInference;



    }


    public void iterateSample(T currentWord, int[] windowWords, boolean[] wordStatuses, AtomicLong nextRandom, double alpha,
                              boolean isInference, int numLabels, boolean trainWords, INDArray inferenceVector) {
        int[] idxSyn1 = null;
        byte[] codes = null;

        if (configuration.isUseHierarchicSoftmax()) {
            idxSyn1 = new int[currentWord.getCodeLength()];
            codes = new byte[currentWord.getCodeLength()];
            for (int p = 0; p < currentWord.getCodeLength(); p++) {
                if (currentWord.getPoints().get(p) < 0)
                    continue;

                codes[p] = currentWord.getCodes().get(p);
                idxSyn1[p] = currentWord.getPoints().get(p);
            }
        } else {
            idxSyn1 = new int[0];
            codes = new byte[0];
        }


        if (negative > 0) {
            if (syn1Neg == null) {
                ((InMemoryLookupTable<T>) lookupTable).initNegative();
                syn1Neg = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1Neg());
            }
        }

        if (batches.get() == null)
            batches.set(new ArrayList<>());



        boolean useHS = configuration.isUseHierarchicSoftmax();
        boolean useNegative = configuration.getNegative() > 0;

        int[] inputStatuses = new int[windowWords.length];
        for (int i = 0; i < windowWords.length; ++i) {
            if (i < wordStatuses.length)
                inputStatuses[i] = wordStatuses[i] ? 1 : 0;
            else
                inputStatuses[i] = -1;
        }



        CbowInference cbowInference = null;
        if (useHS && useNegative) {
            cbowInference = CbowInference.builder()
                    .ngStarter(currentWord.getIndex())
                    .target(currentWord.getIndex())
                    .randomValue((int) nextRandom.get())
                    .nsRounds((int) negative)
                    .syn0(syn0.get())
                    .syn1(syn1.get())
                    .syn1Neg(syn1Neg.get())
                    .expTable(expTable.get())
                    .negTable(table.get() == null ? Nd4j.empty(syn0.get().dataType()) : table.get())
                    .codes(codes)
                    .indices(idxSyn1)
                    .preciseMode(configuration.isPreciseMode())
                    .alpha(alpha)
                    .inferenceVector( inferenceVector != null ? inferenceVector : Nd4j.empty(syn0.get().dataType()))
                    .numWorkers(workers)
                    .build();

        }
        else if (useHS) {
            cbowInference = CbowInference.builder()
                    .target(currentWord.getIndex())
                    .ngStarter(currentWord.getIndex())
                    .randomValue((int) nextRandom.get())
                    .context(windowWords)
                    .expTable(expTable.get())
                    .lockedWords(inputStatuses)
                    .syn0(syn0.get())
                    .syn1(syn1.get())
                    .syn1Neg(Nd4j.empty(syn0.get().dataType()))
                    .negTable(table.get() == null ? Nd4j.empty(syn0.get().dataType()) : table.get())
                    .codes(codes)
                    .indices(new int[0])
                    .numLabels(numLabels)
                    .alpha(alpha)
                    .preciseMode(configuration.isPreciseMode())
                    .inferenceVector(inferenceVector != null ? inferenceVector : Nd4j.empty(syn0.get().dataType()))
                    .numWorkers(workers)
                    .build();

        }
        else if (useNegative) {
            cbowInference = CbowInference.builder()
                    .target(currentWord.getIndex())
                    .ngStarter(currentWord.getIndex())
                    .randomValue((int) nextRandom.get())
                    .context(windowWords)
                    .expTable(expTable.get())
                    .negTable(table.get())
                    .lockedWords(inputStatuses)
                    .syn0(syn0.get())
                    .syn1(Nd4j.empty(syn0.get().dataType()))
                    .syn1Neg(Nd4j.empty(syn0.get().dataType()))
                    .codes(new byte[0])
                    .indices(new int[0])
                    .numLabels(numLabels)
                    .alpha(alpha)
                    .preciseMode(configuration.isPreciseMode())
                    .inferenceVector(inferenceVector != null ? inferenceVector : Nd4j.empty(syn0.get().dataType()))
                    .numWorkers(workers)
                    .build();
        }

        nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
        Nd4j.getExecutioner().exec(cbowInference);



    }




    public void iterateSample(List<BatchItem<T>> items) {

        boolean useHS = configuration.isUseHierarchicSoftmax();
        boolean useNegative = configuration.getNegative() > 0;

        int[] idxSyn1 = null;
        byte[] codes = null;

        int maxCols = 1;
        for (int i = 0; i < items.size(); ++i) {
            int curr = items.get(i).getWord().getCodeLength();
            if (curr > maxCols)
                maxCols = curr;
        }

        byte[][] inputCodes = new byte[items.size()][maxCols];
        int[][] inputIndices = new int[items.size()][maxCols];
        int[] numLabels = new int[items.size()];
        boolean hasNumLabels = false;

        int maxWinWordsCols = -1;
        for (int i = 0; i < items.size(); ++i) {
            int curr = items.get(i).getWindowWords().length;
            if (curr > maxWinWordsCols)
                maxWinWordsCols = curr;
        }
        int[][] inputWindowWords = new int[items.size()][maxWinWordsCols];
        int[][] inputWordsStatuses = new int[items.size()][maxWinWordsCols];

        long[] randoms = new long[items.size()];
        double[] alphas = new double[items.size()];
        int[]  currentWordIndexes = new int[items.size()];

        for (int cnt = 0; cnt < items.size(); ++cnt) {

            T currentWord = items.get(cnt).getWord();
            currentWordIndexes[cnt] = currentWord.getIndex();

            int[] windowWords = items.get(cnt).getWindowWords().clone();
            boolean[] windowStatuses = items.get(cnt).getWordStatuses().clone();

            for (int i = 0; i < maxWinWordsCols; ++i) {
                if (i < windowWords.length) {
                    inputWindowWords[cnt][i] = windowWords[i];
                    inputWordsStatuses[cnt][i] = windowStatuses[i] ? 1 : 0;
                }
                else {
                    inputWindowWords[cnt][i] = -1;
                    inputWordsStatuses[cnt][i] = -1;
                }
            }

            long randomValue = items.get(cnt).getRandomValue();
            double alpha = items.get(cnt).getAlpha();
            alphas[cnt] = alpha;

            randoms[cnt] = randomValue;
            numLabels[cnt] = items.get(cnt).getNumLabel();
            if (numLabels[cnt] > 0)
                hasNumLabels = true;

            if (useHS) {
                idxSyn1 = new int[currentWord.getCodeLength()];
                codes = new byte[currentWord.getCodeLength()];
                for (int p = 0; p < currentWord.getCodeLength(); p++) {
                    if (currentWord.getPoints().get(p) < 0)
                        continue;

                    codes[p] = currentWord.getCodes().get(p);
                    idxSyn1[p] = currentWord.getPoints().get(p);
                }
                for (int i = 0; i < maxCols; ++i) {
                    if (i < currentWord.getCodeLength())
                        inputCodes[cnt][i] = codes[i];
                    else
                        inputCodes[cnt][i] = -1;
                }
                for (int i = 0; i < maxCols; ++i) {
                    if (i < currentWord.getCodeLength())
                        inputIndices[cnt][i]  = idxSyn1[i];
                    else
                        inputIndices[cnt][i] = -1;
                }
            } else {
                idxSyn1 = new int[0];
                codes = new byte[0];

                inputIndices = new int[0][0];
                inputCodes = new byte[0][0];
            }


            if (negative > 0) {
                if (syn1Neg == null) {
                    ((InMemoryLookupTable<T>) lookupTable).initNegative();
                    syn1Neg = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1Neg());
                }
            }

            if (batches.get() == null)
                batches.set(new ArrayList<Aggregate>());

        }

        INDArray currentWordIndexesArray = Nd4j.createFromArray(currentWordIndexes);
        INDArray alphasArray = Nd4j.createFromArray(alphas);
        INDArray windowWordsArray = Nd4j.createFromArray(inputWindowWords);
        INDArray wordsStatusesArray = Nd4j.createFromArray(inputWordsStatuses);
        INDArray codesArray = Nd4j.createFromArray(inputCodes);
        INDArray indicesArray = Nd4j.createFromArray(inputIndices);
        INDArray numLabelsArray = Nd4j.createFromArray(numLabels);

        CbowRound cbow = new CbowRound(currentWordIndexesArray, windowWordsArray, wordsStatusesArray,
                currentWordIndexesArray,
                syn0.get(),
                useHS? syn1.get() : Nd4j.empty(syn0.get().dataType()),
                (negative > 0) ? syn1Neg.get() : Nd4j.empty(syn0.get().dataType()),
                expTable.get(),
                (negative > 0) ? table.get() : Nd4j.empty(syn0.get().dataType()),
                useHS ? indicesArray : Nd4j.empty(DataType.INT),
                useHS ? codesArray : Nd4j.empty(DataType.BYTE),
                (int) negative, alphasArray, Nd4j.createFromArray(randoms),
                /*inferenceVector != null ? inferenceVector :*/ Nd4j.empty(syn0.get().dataType()),
                hasNumLabels ? numLabelsArray : Nd4j.empty(DataType.INT),
                configuration.isTrainElementsVectors(),
                workers);

        Nd4j.getExecutioner().exec(cbow);


    }

    public void cbow(int i, List<T> sentence, int b, AtomicLong nextRandom, double alpha, int currentWindow,
                     BatchSequences<T> batchSequences) {
        int batchSize = configuration.getBatchSize();

        int end = window * 2 + 1 - b;

        T currentWord = sentence.get(i);

        List<Integer> intsList = new ArrayList<>();
        List<Boolean> statusesList = new ArrayList<>();
        for (int a = b; a < end; a++) {
            if (a != currentWindow) {
                int c = i - currentWindow + a;
                if (c >= 0 && c < sentence.size()) {
                    T lastWord = sentence.get(c);

                    intsList.add(lastWord.getIndex());
                    statusesList.add(lastWord.isLocked());
                }
            }
        }

        int[] windowWords = new int[intsList.size()];
        boolean[] statuses = new boolean[intsList.size()];
        for (int x = 0; x < windowWords.length; x++) {
            windowWords[x] = intsList.get(x);
            statuses[x] = statusesList.get(x);
        }

        // we don't allow inference from main loop here
        if (batchSize <= 1)
            iterateSample(currentWord, windowWords, statuses, nextRandom, alpha, false, 0, true, null);
        else {
            batchSequences.put(currentWord, windowWords, statuses, nextRandom.get(), alpha);
        }

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
