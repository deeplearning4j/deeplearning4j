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
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.nlp.SkipGramInference;
import org.nd4j.linalg.api.ops.impl.nlp.SkipGramRound;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.DeviceLocalNDArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

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
    private ArrayCacheMemoryMgr arrayCacheMemoryMgr;
    protected int workers = Runtime.getRuntime().availableProcessors();

    public int getWorkers() {
        return workers;
    }

    public void setWorkers(int workers) {
        this.workers = workers;
    }

    @Getter
    @Setter
    protected DeviceLocalNDArray syn0, syn1, syn1Neg, table, expTable;

    protected ThreadLocal<List<BatchItem<T>>> batches = new ThreadLocal<>();


    /**
     * Dummy construction is required for reflection
     */
    public SkipGram() {
    }

    public List<BatchItem<T>> getBatch() {
        if (batches.get() == null)
            batches.set(new ArrayList<>());
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
                lookupTable.resetWeights(false);
            }
        }

        this.syn0 = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn0());
        this.syn1 = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1());
        this.syn1Neg = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1Neg());
        this.expTable = new DeviceLocalNDArray(Nd4j.create(((InMemoryLookupTable<T>) lookupTable).getExpTable(),
                new long[]{((InMemoryLookupTable<T>) lookupTable).getExpTable().length}, syn0.get() == null ? DataType.DOUBLE
                        : syn0.get().dataType()));
        this.table = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getTable());

        arrayCacheMemoryMgr = new ArrayCacheMemoryMgr();

        this.window = configuration.getWindow();
        this.useAdaGrad = configuration.isUseAdaGrad();
        this.negative = configuration.getNegative();
        this.sampling = configuration.getSampling();
        this.variableWindows = configuration.getVariableWindows();
        this.workers = configuration.getWorkers();
        this.vectorLength = configuration.getLayersSize();
    }

    /**
     * SkipGram doesn't involve any pretraining
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
            currentWindow = variableWindows[RandomUtils.nextInt(0, variableWindows.length)];
        }
        for (int i = 0; i < tempSequence.getElements().size(); i++) {
            nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
            score = skipGram(i, tempSequence.getElements(), (int) nextRandom.get() % currentWindow, nextRandom,
                    learningRate, currentWindow);
        }

        if (getBatch() != null && getBatch().size() >= configuration.getBatchSize()) {
            score = iterateSample(null, null);
            getBatch().clear();
        }

        return score;
    }

    @Override
    public void finish() {
        if (batches != null && batches.get() != null && !batches.get().isEmpty()) {
            iterateSample(null, null);
            batches.get().clear();
        }
    }

    @Override
    public void finish(INDArray inferenceVector) {
        if (batches != null && batches.get() != null && !batches.get().isEmpty()) {
            iterateSample(null, inferenceVector);
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
        if (word == null || sentence.isEmpty() || word.isLocked())
            return 0.0;

        double score = 0.0;
        int end = currentWindow * 2 + 1 - b;
        for (int a = b; a < end; a++) {
            if (a != currentWindow) {
                int c = i - currentWindow + a;
                if (c >= 0 && c < sentence.size()) {
                    T lastWord = sentence.get(c);
                    nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
                    BatchItem<T> batchItem = new BatchItem<>(word, lastWord, nextRandom.get(), alpha);
                    score = iterateSample(batchItem,null);

                }
            }
        }

        return score;
    }


    public double iterateSample(BatchItem<T> item,INDArray inferenceVector) {
        double score = 0.0;

        List<BatchItem<T>> items = getBatch();
        if(item != null) {
            items.add(item);
            if(items.size() >= configuration.getBatchSize()) {
                score = doExec(items, null);
            }
        } else if(item == null && !items.isEmpty()) {
            if(items.size() >= configuration.getBatchSize()) {
                score = doExec(items, null);
            }
        }

        return score;

    }

    public  Double doExec(List<BatchItem<T>> items,INDArray inferenceVector) {
        if (items.size() > 1) {
            INDArray targetArray = arrayCacheMemoryMgr.allocate(false, DataType.INT32, items.size());
            INDArray ngStarterArray = arrayCacheMemoryMgr.allocate(false, DataType.INT32, items.size());
            INDArray alphasArray = arrayCacheMemoryMgr.allocate(false, DataType.DOUBLE, items.size());
            INDArray randomValuesArr = arrayCacheMemoryMgr.allocate(false, DataType.INT64, items.size());
            int maxCols = 1;
            for (int i = 0; i < items.size(); i++) {
                int curr = items.get(i).getWord().getCodeLength();
                if (curr > maxCols)
                    maxCols = curr;
            }


            //use -1 as padding for codes that are not actually valid for a given row
            INDArray codes = Nd4j.valueArrayOf(new long[]{items.size(),maxCols},-1,DataType.INT32);
            INDArray indices = Nd4j.create(DataType.INT32, items.size(), maxCols);

            for (int cnt = 0; cnt < items.size(); cnt++) {

                T w1 = items.get(cnt).getWord();
                T lastWord = items.get(cnt).getLastWord();

                randomValuesArr.putScalar(cnt, items.get(cnt).getRandomValue());
                double alpha = items.get(cnt).getAlpha();

                if (w1 == null || lastWord == null || (lastWord.getIndex() < 0 && inferenceVector == null)

                        || w1.getIndex() == lastWord.getIndex() || w1.getLabel().equals("STOP")
                        || lastWord.getLabel().equals("STOP") || w1.getLabel().equals("UNK")
                        || lastWord.getLabel().equals("UNK")) {
                    continue;
                }


                int target = lastWord.getIndex();
                int ngStarter = w1.getIndex();


                targetArray.putScalar(cnt, target);
                ngStarterArray.putScalar(cnt, ngStarter);
                alphasArray.putScalar(cnt, alpha);


                if (configuration.isUseHierarchicSoftmax()) {
                    for (int i = 0; i < w1.getCodeLength(); i++) {
                        int code = w1.getCodes().get(i);
                        int point = w1.getPoints().get(i);
                        if (point >= vocabCache.numWords() || point < 0)
                            continue;
                        codes.putScalar(cnt, i, code);
                        indices.putScalar(cnt, i, point);
                    }

                }

                //negative sampling
                if (negative > 0) {
                    if (syn1Neg == null) {
                        ((InMemoryLookupTable<T>) lookupTable).initNegative();
                        syn1Neg = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1Neg());
                    }
                }
            }




            SkipGramRound sg = SkipGramRound.builder()
                    .target(targetArray)
                    .expTable(expTable.get())
                    .ngStarter((negative > 0) ? ngStarterArray : Nd4j.empty(DataType.INT32))
                    .syn0(syn0.get())
                    .syn1(configuration.isUseHierarchicSoftmax() ? syn1.get() : Nd4j.empty(syn0.get().dataType()))
                    .syn1Neg((negative > 0) ? syn1Neg.get() : Nd4j.empty(syn0.get().dataType()))
                    .negTable((negative > 0) ? table.get() : Nd4j.empty(syn0.get().dataType()))
                    .indices(configuration.isUseHierarchicSoftmax() ? indices : Nd4j.empty(DataType.INT32))
                    .codes(configuration.isUseHierarchicSoftmax() ? codes: Nd4j.empty(DataType.INT8))

                    .alpha(alphasArray)
                    .randomValue(randomValuesArr)
                    .inferenceVector(inferenceVector != null ? inferenceVector : Nd4j.empty(syn0.get().dataType()))
                    .preciseMode(configuration.isPreciseMode())
                    .numWorkers(workers)
                    .iterations(inferenceVector != null ? configuration.getIterations() * configuration.getEpochs() : 1)
                    .build();

            Nd4j.getExecutioner().exec(sg);
            arrayCacheMemoryMgr.release(codes);
            arrayCacheMemoryMgr.release(indices);
            arrayCacheMemoryMgr.release(targetArray);
            arrayCacheMemoryMgr.release(randomValuesArr);
            arrayCacheMemoryMgr.release(alphasArray);
            items.clear();


        } else {
            int cnt = 0;

            T w1 = items.get(cnt).getWord();
            T lastWord = items.get(cnt).getLastWord();
            byte[] codes = new byte[w1.getCodeLength()];
            int[] indices = new int[w1.getCodeLength()];

            double alpha = items.get(cnt).getAlpha();

            if (w1 == null || lastWord == null || (lastWord.getIndex() < 0 && inferenceVector == null)
                    || w1.getIndex() == lastWord.getIndex() || w1.getLabel().equals("STOP")
                    || lastWord.getLabel().equals("STOP") || w1.getLabel().equals("UNK")
                    || lastWord.getLabel().equals("UNK")) {
                return 0.0;
            }

            int target = lastWord.getIndex();
            int ngStarter = w1.getIndex();


            if (configuration.isUseHierarchicSoftmax()) {

                for (int i = 0; i < w1.getCodeLength(); i++) {
                    int code = w1.getCodes().get(i);
                    int point = w1.getPoints().get(i);
                    if (point >= vocabCache.numWords() || point < 0)
                        continue;
                    if (i < w1.getCodeLength()) {
                        codes[i] = (byte) code;
                        indices[i] = point;
                    }

                }

            }

            //negative sampling
            if (negative > 0) {
                if (syn1Neg == null) {
                    ((InMemoryLookupTable<T>) lookupTable).initNegative();
                    syn1Neg = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1Neg());
                }
            }


            SkipGramInference sg = SkipGramInference.builder()
                    .inferenceVector(inferenceVector != null ? inferenceVector : Nd4j.empty(syn0.get().dataType()))
                    .randomValue((int) items.get(0).getRandomValue())
                    .syn0(syn0.get())
                    .negTable((negative > 0) ? table.get() : Nd4j.empty(syn0.get().dataType()))
                    .expTable(expTable.get())
                    .syn1(configuration.isUseHierarchicSoftmax() ? syn1.get() : Nd4j.empty(syn0.get().dataType()))
                    .syn1Neg((negative > 0) ? syn1Neg.get() : Nd4j.empty(syn0.get().dataType()))
                    .negTable((negative > 0) ? table.get() : Nd4j.empty(syn0.get().dataType()))
                    .alpha(new double[]{alpha})
                    .iteration(1)

                    .ngStarter(ngStarter)
                    .indices(indices)
                    .target(target)
                    .codes(codes)
                    .preciseMode(configuration.getPreciseMode())
                    .numWorkers(configuration.getWorkers())
                    .build();

            Nd4j.getExecutioner().exec(sg);
            items.clear();

        }
        return 0.0;

    }
}
