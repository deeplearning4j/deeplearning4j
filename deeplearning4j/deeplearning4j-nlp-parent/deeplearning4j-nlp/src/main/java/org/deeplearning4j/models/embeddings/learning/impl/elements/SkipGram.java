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
import lombok.val;
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
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.api.ops.impl.nlp.SkipGramRound;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.DeviceLocalNDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

@Slf4j
public class SkipGram<T extends SequenceElement> implements ElementsLearningAlgorithm<T> {
    protected VocabCache<T> vocabCache;
    protected WeightLookupTable<T> lookupTable;
    protected VectorsConfiguration configuration;
    private ArrayCacheMemoryMgr  arrayCacheMemoryMgr;
    protected int window;
    protected boolean useAdaGrad;
    protected double negative;
    protected double sampling;
    protected int[] variableWindows;
    protected int vectorLength;
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
        arrayCacheMemoryMgr = new ArrayCacheMemoryMgr();
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



        this.window = configuration.getWindow();
        this.useAdaGrad = configuration.isUseAdaGrad();
        this.negative = configuration.getNegative();
        this.sampling = configuration.getSampling();
        this.variableWindows = configuration.getVariableWindows();

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

    public double learnSequence(@NonNull Sequence<T> sequence, @NonNull AtomicLong nextRandom, double learningRate,
                                BatchSequences<T> batchSequences) {
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
            if(currentWindow > 0)
                score = skipGram(i, tempSequence.getElements(), (int) nextRandom.get() % currentWindow, nextRandom,
                        learningRate, currentWindow, batchSequences);
            else
                score = skipGram(i, tempSequence.getElements(), (int) nextRandom.get() , nextRandom,
                        learningRate, currentWindow, batchSequences);

        }


        if (batches != null && batches.get() != null && batches.get().size() >= configuration.getBatchSize()) {
            Nd4j.getExecutioner().exec(batches.get());
            batches.get().clear();
        }

        return score;
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
        //batchSequences = new BatchSequences<>(configuration.getBatchSize());
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

    private double skipGram(int i, List<T> sentence, int b, AtomicLong nextRandom, double alpha, int currentWindow,
                            BatchSequences<T> batchSequences) {
        final T word = sentence.get(i);
        if (word == null || sentence.isEmpty() || word.isLocked())
            return 0.0;

        double score = 0.0;
        int batchSize = configuration.getBatchSize();

        int end = currentWindow * 2 + 1 - b;
        for (int a = b; a < end; a++) {
            if (a != currentWindow) {
                int c = i - currentWindow + a;
                if (c >= 0 && c < sentence.size()) {
                    T lastWord = sentence.get(c);
                    nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
                    if (batchSize <= 1) {
                        score = iterateSample(word, lastWord, nextRandom, alpha, false, null);
                    }
                    else {
                        batchSequences.put(word, lastWord, nextRandom.get(), alpha);
                    }
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
        byte[] codes = null;
        if (configuration.isUseHierarchicSoftmax()) {
            idxSyn1 = new int[w1.getCodeLength()];
            codes = new byte[w1.getCodeLength()];
            for (int i = 0; i < w1.getCodeLength(); i++) {
                int code = w1.getCodes().get(i);
                int point = w1.getPoints().get(i);
                if (point >= vocabCache.numWords() || point < 0)
                    continue;

                codes[i] = (byte)code;
                idxSyn1[i] = point;
            }
        } else {
            idxSyn1 = new int[0];
            codes = new byte[0];
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
            batches.set(new ArrayList<>());
        }

        nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));

        SkipGramRound sg = null;
        boolean useHS = configuration.isUseHierarchicSoftmax();
        boolean useNegative = configuration.getNegative() > 0;

        byte[] intCodes = new byte[codes.length];
        for (int i = 0; i < codes.length; ++i) {
            intCodes[i] = codes[i];
        }

        List<INDArray> release = new ArrayList<>();
        if (useHS && useNegative) {
            INDArray allocate = arrayCacheMemoryMgr.allocate(false,DataType.INT32,1);
            INDArray targetArr = arrayCacheMemoryMgr.allocate(false,DataType.INT32,1);
            INDArray alphaArr = arrayCacheMemoryMgr.allocate(false,syn0.get().dataType(),1);
            INDArray nextRandomArr = arrayCacheMemoryMgr.allocate(true,DataType.INT64,1);
            allocate.putScalar(0,lastWord.getIndex());
            targetArr.putScalar(0,target);
            alphaArr.putScalar(0,alpha);
            if(nextRandom != null)
                nextRandomArr.putScalar(0,nextRandom.get());
            INDArray idxSyn1Arr = arrayCacheMemoryMgr.allocate(false,DataType.INT32,idxSyn1.length);

          sg = SkipGramRound.builder()
                  .target(allocate)
                  .ngStarter(targetArr)
                  .syn0(syn0.get())
                  .syn1(syn1.get())
                  .syn1Neg(syn1Neg.get())
                  .expTable(expTable.get())
                  .negTable(table.get())
                  .nsRounds(0)
                  .indices(idxSyn1Arr)
                  .codes(Nd4j.createFromArray(intCodes))
                  .randomValue(nextRandomArr)
                  .inferenceVector(inferenceVector != null ? inferenceVector : Nd4j.empty(syn0.get().dataType()))
                  .alpha(alphaArr)
                  .preciseMode(configuration.isPreciseMode())
                  .numWorkers(workers)
                  .build();

            release.add(alphaArr);
            release.add(targetArr);
            release.add(idxSyn1Arr);
            release.add(allocate);

        }
        else if (useHS) {
            INDArray lastWordIdx = arrayCacheMemoryMgr.allocate(false,DataType.INT32,1);
            INDArray targetArr = arrayCacheMemoryMgr.allocate(false,DataType.INT32,1);
            INDArray alphaArr = arrayCacheMemoryMgr.allocate(false,syn0.get().dataType(),1);
            INDArray nextRandomArr = arrayCacheMemoryMgr.allocate(false,DataType.INT64,1);
            INDArray ngStarter = arrayCacheMemoryMgr.allocate(false,DataType.INT32,1);
            ngStarter.putScalar(0,-1);
            lastWordIdx.putScalar(0,lastWord.getIndex());
            targetArr.putScalar(0,target);
            alphaArr.putScalar(0,alpha);
            nextRandomArr.putScalar(0,nextRandom.get());

            sg = SkipGramRound.builder()
                    .target(lastWordIdx)
                    .ngStarter(targetArr)
                    .syn0(syn0.get())
                    .syn1(syn1.get())
                    .syn1Neg(Nd4j.empty(syn0.get().dataType()))
                    .expTable(expTable.get() != null ? expTable.get() : Nd4j.empty(syn0.get().dataType()))
                    .negTable(table.get() != null ? table.get() : Nd4j.empty(syn0.get().dataType()))
                    .nsRounds((int) negative)
                    .alpha(alphaArr)
                    .randomValue(nextRandomArr)
                    .inferenceVector(inferenceVector != null ? inferenceVector : Nd4j.empty(syn0.get().dataType()))
                    .codes( Nd4j.createFromArray(intCodes))
                    .indices(Nd4j.createFromArray(idxSyn1))
                    .preciseMode(configuration.isPreciseMode())
                    .numWorkers(workers)
                    .build();


            release.add(lastWordIdx);
            release.add(alphaArr);
            release.add(targetArr);
            release.add(ngStarter);
            release.add(nextRandomArr);

        }
        else if (useNegative) {
            INDArray lastWordIdx = arrayCacheMemoryMgr.allocate(false,DataType.INT32,1);
            INDArray targetArr = arrayCacheMemoryMgr.allocate(false,DataType.INT32,1);
            INDArray alphaArr = arrayCacheMemoryMgr.allocate(false,syn0.get().dataType(),1);
            INDArray nextRandomArr = arrayCacheMemoryMgr.allocate(false,DataType.INT64,1);
            INDArray ngStarter = arrayCacheMemoryMgr.allocate(false,DataType.INT32,1);
            INDArray negativeArr = arrayCacheMemoryMgr.allocate(false,DataType.INT32,1);
            negativeArr.putScalar(0,negative);
            ngStarter.putScalar(0,target);
            lastWordIdx.putScalar(0,lastWord.getIndex());
            targetArr.putScalar(0,target);
            alphaArr.putScalar(0,alpha);
            nextRandomArr.putScalar(0,nextRandom.get());

            //    int numWorkers) {
            sg = SkipGramRound.builder()
                    .ngStarter(targetArr)
                    .target(lastWordIdx)
                    .syn0(syn0.get())
                    .syn1(Nd4j.empty(syn0.get().dataType()))
                    .codes(Nd4j.empty(DataType.INT8))
                    .indices(Nd4j.empty(DataType.INT32))
                    .negTable(negativeArr)
                    .syn1Neg(syn1Neg.get())
                    .expTable(expTable.get())
                    .syn1(syn1.get())
                    .alpha(alphaArr)
                    .preciseMode(configuration.isPreciseMode())
                    .numWorkers(workers)
                    .inferenceVector(inferenceVector != null ? inferenceVector : Nd4j.empty(syn0.get().dataType()))
                    .build();

            release.add(negativeArr);
            release.add(lastWordIdx);
            release.add(alphaArr);
            release.add(targetArr);
            release.add(ngStarter);
            release.add(nextRandomArr);

        }

        Nd4j.getExecutioner().exec(sg);

        //release after use
        for(INDArray arr : release)
            arrayCacheMemoryMgr.release(arr);

        return score;
    }

    public double iterateSample(List<BatchItem<T>> items) {
        boolean useHS = configuration.isUseHierarchicSoftmax();
        boolean useNegative = configuration.getNegative() > 0;

        double score = 0.0;
        boolean isInference = false;

        int[] targets = new int[items.size()];
        int[] starters = new int[items.size()];
        double[] alphas = new double[items.size()];
        long[] randomValues = new long[items.size()];

        int maxCols = 1;
        for (int i = 0; i < items.size(); ++i) {
            int curr = items.get(i).getWord().getCodeLength();
            if (curr > maxCols)
                maxCols = curr;
        }

        byte[][] codes = new byte[items.size()][maxCols];
        int[][] indices = new int[items.size()][maxCols];

        for (int cnt = 0; cnt < items.size(); ++cnt) {
            T w1 = items.get(cnt).getWord();
            T lastWord = items.get(cnt).getLastWord();
            randomValues[cnt] = items.get(cnt).getRandomValue();
            double alpha = items.get(cnt).getAlpha();

            if (w1 == null || lastWord == null || (lastWord.getIndex() < 0 && !isInference)
                    || w1.getIndex() == lastWord.getIndex() || w1.getLabel().equals("STOP")
                    || lastWord.getLabel().equals("STOP") || w1.getLabel().equals("UNK")
                    || lastWord.getLabel().equals("UNK")) {
                continue;
            }

            int target = lastWord.getIndex();
            int ngStarter = w1.getIndex();

            targets[cnt] = target;
            starters[cnt] = ngStarter;
            alphas[cnt] = alpha;

            int[] idxSyn1 = null;
            byte[] interimCodes = null;
            if (useHS) {
                idxSyn1 = new int[w1.getCodeLength()];
                interimCodes = new byte[w1.getCodeLength()];
                for (int i = 0; i < w1.getCodeLength(); i++) {
                    int code = w1.getCodes().get(i);
                    int point = w1.getPoints().get(i);
                    if (point >= vocabCache.numWords() || point < 0)
                        continue;

                    interimCodes[i] = (byte) code;
                    idxSyn1[i] = point;
                }
                for (int i = 0; i < maxCols; ++i) {
                    if (i < w1.getCodeLength())
                        codes[cnt][i] = interimCodes[i];
                    else
                        codes[cnt][i] = -1;
                }
                for (int i = 0; i < maxCols; ++i) {
                    if (i < w1.getCodeLength())
                        indices[cnt][i]  = idxSyn1[i];
                    else
                        indices[cnt][i] = -1;
                }

            } else {
                idxSyn1 = new int[0];
                interimCodes = new byte[0];
                codes = new byte[0][0];
                indices = new int[0][0];
            }

            //negative sampling
            if (negative > 0) {
                if (syn1Neg == null) {
                    ((InMemoryLookupTable<T>) lookupTable).initNegative();
                    syn1Neg = new DeviceLocalNDArray(((InMemoryLookupTable<T>) lookupTable).getSyn1Neg());
                }
            }
        }


        INDArray targetArray = Nd4j.createFromArray(targets);
        INDArray ngStarterArray = Nd4j.createFromArray(starters);
        INDArray alphasArray = Nd4j.createFromArray(alphas);
        INDArray randomValuesArray = Nd4j.createFromArray(randomValues);
        INDArray indicesArray = Nd4j.createFromArray(indices);
        INDArray codesArray = Nd4j.createFromArray(codes);
        SkipGramRound sg = SkipGramRound.builder()
                .target(targetArray)
                .expTable(expTable.get())
                .ngStarter( (negative > 0) ? ngStarterArray : Nd4j.empty(DataType.INT32))
                .syn0(syn0.get())
                .syn1(useHS ? syn1.get() : Nd4j.empty(syn0.get().dataType()))
                .syn1Neg( (negative > 0) ? syn1Neg.get() : Nd4j.empty(syn0.get().dataType()))
                .negTable( (negative > 0) ? table.get() : Nd4j.empty(syn0.get().dataType()))
                .indices( useHS ? indicesArray : Nd4j.empty(DataType.INT32))
                .codes(useHS ? codesArray : Nd4j.empty(DataType.INT8))
                .alpha(alphasArray)
                .randomValue(randomValuesArray)
                .inferenceVector( Nd4j.empty(syn0.get().dataType()))
                .preciseMode(configuration.isPreciseMode())
                .numWorkers(workers)
                .build();



        Nd4j.getExecutioner().exec(sg);

        return score;
    }
}