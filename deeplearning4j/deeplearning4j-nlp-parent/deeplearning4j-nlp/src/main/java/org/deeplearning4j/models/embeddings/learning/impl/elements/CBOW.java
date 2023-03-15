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

import lombok.*;
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
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.nlp.CbowInference;
import org.nd4j.linalg.api.ops.impl.nlp.CbowRound;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.DeviceLocalNDArray;
import org.nd4j.shade.guava.cache.Cache;
import org.nd4j.shade.guava.cache.CacheBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

public class CBOW<T extends SequenceElement> implements ElementsLearningAlgorithm<T> {
    private VocabCache<T> vocabCache;
    private WeightLookupTable<T> lookupTable;
    private VectorsConfiguration configuration;

    private static final Logger logger = LoggerFactory.getLogger(CBOW.class);


    protected int window;
    protected boolean useAdaGrad;
    protected double negative;
    protected double sampling;
    protected int[] variableWindows;
    protected int workers = Runtime.getRuntime().availableProcessors();
    private Cache<IterationArraysKey, Queue<IterationArrays>> iterationArrays = CacheBuilder.newBuilder().build();

    public int getWorkers() {
        return workers;
    }

    public void setWorkers(int workers) {
        this.workers = workers;
    }

    @Getter
    @Setter
    protected DeviceLocalNDArray syn0, syn1, syn1Neg, expTable, table;

    protected ThreadLocal<List<BatchItem<T>>> batches = new ThreadLocal<>();

    public List<BatchItem<T>> getBatch() {
        if(batches.get() == null)
            batches.set(new ArrayList<>());
        return batches.get();
    }


    public void addBatchItem(BatchItem<T> batchItem) {
        getBatch().add(batchItem);
    }


    public void clearBatch() {
        getBatch().clear();
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
        this.workers = configuration.getWorkers();
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
            doExec(batches.get(),null);
            batches.get().clear();
        }
    }

    @Override
    public void finish(INDArray inferenceVector) {
        if (batches != null && batches.get() != null && !batches.get().isEmpty()) {
            doExec(batches.get(),inferenceVector);
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
            currentWindow = variableWindows[RandomUtils.nextInt(0, variableWindows.length)];
        }

        for (int i = 0; i < tempSequence.getElements().size(); i++) {
            nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
            cbow(i, tempSequence.getElements(), (int) nextRandom.get() % currentWindow, nextRandom, learningRate,
                    currentWindow, null);
        }

        if (getBatch() != null && getBatch().size() >= configuration.getBatchSize()) {
            doExec(getBatch(),null);
            getBatch().clear();
        }


        return 0;
    }

    @Override
    public boolean isEarlyTerminationHit() {
        return false;
    }

    @Data
    @AllArgsConstructor
    @Builder
    @NoArgsConstructor
    public static class IterationArraysKey {
        private int itemSize;
        private int maxCols;
    }


    public double doExec(List<BatchItem<T>> items, INDArray inferenceVector) {
        try(MemoryWorkspace workspace = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            boolean useHS = configuration.isUseHierarchicSoftmax();
            boolean useNegative = configuration.getNegative() > 0;
            boolean useInference = inferenceVector != null;
            if (items.size() > 1) {
                int maxCols = 1;
                for (int i = 0; i < items.size(); i++) {
                    int curr = items.get(i).getWord().getCodeLength();
                    if (curr > maxCols)
                        maxCols = curr;
                }


                boolean hasNumLabels = false;

                int maxWinWordsCols = -1;
                for (int i = 0; i < items.size(); ++i) {
                    int curr = items.get(i).getWord().getCodeLength();
                    if (curr > maxWinWordsCols)
                        maxWinWordsCols = curr;
                }





                INDArray inputWindowWords;
                INDArray inputWordsStatuses;
                INDArray randoms;
                INDArray alphas;
                INDArray currentWindowIndexes;
                INDArray codes;
                INDArray indices;
                INDArray numLabelsArray;


                IterationArraysKey key = IterationArraysKey.builder()
                        .itemSize(items.size())
                        .maxCols(maxCols).build();
                Queue<IterationArrays> iterationArraysQueue = iterationArrays.getIfPresent(key);
                IterationArrays iterationArrays1;
                if(iterationArraysQueue == null) {
                    iterationArraysQueue = new ConcurrentLinkedQueue<>();
                    iterationArrays.put(key,iterationArraysQueue);
                    iterationArrays1 = new IterationArrays(items.size(),maxCols);
                } else {
                    if(iterationArraysQueue.isEmpty()) {
                        iterationArrays1 = new IterationArrays(items.size(),maxCols);

                    }else {
                        iterationArrays1 = iterationArraysQueue.remove();
                        iterationArrays1.initCodes();
                    }
                }

                int[][] inputWindowWordsArr = iterationArrays1.inputWindowWordsArr;
                int[][] inputWindowWordStatuses = iterationArrays1.inputWindowWordStatuses;
                int[]  currentWindowIndexesArr = iterationArrays1.currentWindowIndexes;
                double[] alphasArr = iterationArrays1.alphas;
                int[][] indicesArr = iterationArrays1.indicesArr;
                int[][]  codesArr = iterationArrays1.codesArr;
                long[] randomValues = iterationArrays1.randomValues;
                int[] numLabelsArr = iterationArrays1.numLabels;
                currentWindowIndexes = Nd4j.createFromArray(currentWindowIndexesArr);

                for (int cnt = 0; cnt < items.size(); cnt++) {
                    T currentWord = items.get(cnt).getWord();
                    currentWindowIndexes.putScalar(0, currentWord.getIndex());
                    currentWindowIndexesArr[0] = currentWord.getIndex();
                    int[] windowWords = items.get(cnt).getWindowWords().clone();
                    boolean[] windowStatuses = items.get(cnt).getWordStatuses().clone();

                    for (int i = 0; i < maxWinWordsCols; i++) {
                        if (i < windowWords.length) {
                            inputWindowWordsArr[cnt][i] = windowWords[i];
                            inputWindowWordStatuses[cnt][i] = windowStatuses[i] ? 1 : 0;

                        } else {
                            inputWindowWordsArr[cnt][i] = -1;
                            inputWindowWordStatuses[cnt][i] = -1;
                        }
                    }

                    long randomValue = items.get(cnt).getRandomValue();
                    double alpha = items.get(cnt).getAlpha();
                    alphasArr[cnt] = alpha;
                    randomValues[cnt] = randomValue;
                    numLabelsArr[cnt] = items.get(cnt).getNumLabel();
                    if (items.get(cnt).getNumLabel() > 0)
                        hasNumLabels = true;

                    if (useHS) {
                        for (int p = 0; p < currentWord.getCodeLength(); p++) {
                            if (currentWord.getPoints().get(p) < 0)
                                continue;


                            codesArr[cnt][p] = currentWord.getCodes().get(p);
                            indicesArr[cnt][p] =  currentWord.getPoints().get(p);
                        }

                    }

                    if (negative > 0) {
                        if (syn1Neg == null) {
                            ((InMemoryLookupTable<T>) lookupTable).initNegative();
                            syn1Neg = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1Neg());
                        }
                    }

                }


                inputWindowWords = Nd4j.createFromArray(inputWindowWordsArr);
                inputWordsStatuses = Nd4j.createFromArray(inputWindowWordStatuses);
                numLabelsArray = Nd4j.createFromArray(numLabelsArr);
                indices = Nd4j.createFromArray(indicesArr);
                codes = Nd4j.createFromArray(codesArr);
                alphas = Nd4j.createFromArray(alphasArr);
                randoms = Nd4j.createFromArray(randomValues);
                CbowRound cbow = CbowRound.builder()
                        .target(currentWindowIndexes)
                        .context(inputWindowWords)
                        .lockedWords(inputWordsStatuses)
                        .ngStarter(currentWindowIndexes)
                        .syn0(syn0.get())
                        .syn1(useHS ? syn1.get() : Nd4j.empty(syn0.get().dataType()))
                        .syn1Neg((negative > 0) ? syn1Neg.get() : Nd4j.empty(syn0.get().dataType()))
                        .expTable(expTable.get())
                        .negTable(negative > 0 ? table.get() : Nd4j.empty(syn0.get().dataType()))
                        .indices(useHS ? indices : Nd4j.empty(DataType.INT32))
                        .codes(useHS ? codes : Nd4j.empty(DataType.INT8))
                        .nsRounds((int) negative)
                        .alpha(alphas)
                        .nextRandom(randoms)
                        .inferenceVector(inferenceVector != null ? inferenceVector : Nd4j.empty(syn0.get().dataType()))
                        .numLabels(hasNumLabels ? numLabelsArray : Nd4j.empty(DataType.INT32))
                        .trainWords(configuration.isTrainElementsVectors())
                        .numWorkers(workers)
                        .iterations(useInference ? configuration.getIterations() * configuration.getEpochs() : 1)
                        .build();


                Nd4j.getExecutioner().exec(cbow);


                Nd4j.close(currentWindowIndexes,inputWindowWords,alphas,randoms,codes,numLabelsArray,indices);
                iterationArraysQueue.add(iterationArrays1);

                batches.get().clear();
                return 0.0;
            } else {
                int cnt = 0;
                T currentWord = items.get(cnt).getWord();
                int currentWindowIndex = currentWord.getIndex();

                int[] windowWords = items.get(cnt).getWindowWords().clone();
                boolean[] windowStatuses = items.get(cnt).getWordStatuses().clone();
                byte[] codes = new byte[currentWord.getCodeLength()];
                int[] points = new int[currentWord.getCodeLength()];
                long randomValue = items.get(cnt).getRandomValue();
                double alpha = items.get(cnt).getAlpha();
                int numLabels = items.get(cnt).getNumLabel();
                int[] inputStatuses = new int[windowWords.length];
                for (int i = 0; i < windowWords.length; ++i) {
                    if (i < windowStatuses.length)
                        inputStatuses[i] = windowStatuses[i] ? 1 : 0;
                    else
                        inputStatuses[i] = -1;
                }

                if (useHS) {
                    for (int p = 0; p < currentWord.getCodeLength(); p++) {
                        if (currentWord.getPoints().get(p) < 0)
                            continue;

                        codes[p] = currentWord.getCodes().get(p);
                        points[p] = currentWord.getPoints().get(p);
                    }

                }

                if (negative > 0) {
                    if (syn1Neg == null) {
                        ((InMemoryLookupTable<T>) lookupTable).initNegative();
                        syn1Neg = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1Neg());
                    }
                } else {
                    syn1Neg.set(Nd4j.empty(syn0.get().dataType()));
                }

                CbowInference cbowInference = CbowInference.builder()
                        .target(currentWord.getIndex())
                        .ngStarter(currentWord.getIndex())
                        .negTable(table.get() == null ? Nd4j.empty(syn0.get().dataType()) : table.get())
                        .expTable(expTable.get() == null ? Nd4j.empty(syn0.get().dataType()) : expTable.get())
                        .syn0(syn0.get())
                        .syn1(!useHS ? Nd4j.empty(syn0.get().dataType()) : syn1.get())
                        .syn1Neg(syn1Neg.get())
                        .alpha(alpha)
                        .context(windowWords == null ? new int[0] : windowWords)
                        .indices(useHS || useNegative ? points : new int[0])
                        .codes(useHS ? codes : new byte[0])
                        .lockedWords(inputStatuses)
                        .randomValue((int) randomValue)
                        .numWorkers(workers)
                        .numLabels(numLabels)
                        .nsRounds(useNegative ? (int) negative : 0)
                        .preciseMode(configuration.isPreciseMode())
                        .inferenceVector(inferenceVector != null ? inferenceVector : Nd4j.empty(syn0.get().dataType()))
                        .iterations(useInference ? configuration.getIterations() * configuration.getEpochs() : 1)
                        .build();


                Nd4j.getExecutioner().exec(cbowInference);
            }

        }
        return 0.0;
    }

    public void cbow(int i, List<T> sentence, int b, AtomicLong nextRandom, double alpha, int currentWindow,
                     List<BatchItem<T>> batch) {

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

        BatchItem<T> batchItem = new BatchItem<>(currentWord,windowWords,statuses,nextRandom.get(),alpha);
        batch.add(batchItem);
        iterateBatchesIfReady(batch);


    }

    private double iterateBatchesIfReady(List<BatchItem<T>> batch) {
        double score = 0.0;
        if(batches.get() == null) {
            batches.set(batch);
        }
        else
            batches.get().addAll(batch);

        if(batches.get().size() >= configuration.getBatchSize()) {
            score = doExec(batches.get(),null);
            batches.get().clear();

        }
        return score;
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
