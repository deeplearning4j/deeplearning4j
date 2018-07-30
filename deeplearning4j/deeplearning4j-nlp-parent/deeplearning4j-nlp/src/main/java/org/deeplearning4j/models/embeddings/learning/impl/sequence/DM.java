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

package org.deeplearning4j.models.embeddings.learning.impl.sequence;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.learning.SequenceLearningAlgorithm;
import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * DM implementation for DeepLearning4j
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class DM<T extends SequenceElement> implements SequenceLearningAlgorithm<T> {
    private VocabCache<T> vocabCache;
    private WeightLookupTable<T> lookupTable;
    private VectorsConfiguration configuration;

    protected static double MAX_EXP = 6;

    protected int window;
    protected boolean useAdaGrad;
    protected double negative;
    protected double sampling;

    protected double[] expTable;

    protected INDArray syn0, syn1, syn1Neg, table;

    private CBOW<T> cbow = new CBOW<>();

    @Override
    public ElementsLearningAlgorithm<T> getElementsLearningAlgorithm() {
        return cbow;
    }

    @Override
    public String getCodeName() {
        return "PV-DM";
    }

    @Override
    public void configure(@NonNull VocabCache<T> vocabCache, @NonNull WeightLookupTable<T> lookupTable,
                    @NonNull VectorsConfiguration configuration) {
        this.vocabCache = vocabCache;
        this.lookupTable = lookupTable;
        this.configuration = configuration;

        cbow.configure(vocabCache, lookupTable, configuration);

        this.window = configuration.getWindow();
        this.useAdaGrad = configuration.isUseAdaGrad();
        this.negative = configuration.getNegative();
        this.sampling = configuration.getSampling();

        this.syn0 = ((InMemoryLookupTable<T>) lookupTable).getSyn0();
        this.syn1 = ((InMemoryLookupTable<T>) lookupTable).getSyn1();
        this.syn1Neg = ((InMemoryLookupTable<T>) lookupTable).getSyn1Neg();
        this.expTable = ((InMemoryLookupTable<T>) lookupTable).getExpTable();
        this.table = ((InMemoryLookupTable<T>) lookupTable).getTable();
    }

    @Override
    public void pretrain(SequenceIterator<T> iterator) {
        // no-op
    }

    @Override
    public double learnSequence(Sequence<T> sequence, AtomicLong nextRandom, double learningRate) {
        Sequence<T> seq = cbow.applySubsampling(sequence, nextRandom);

        if (sequence.getSequenceLabel() == null)
            return 0;

        List<T> labels = new ArrayList<>();
        labels.addAll(sequence.getSequenceLabels());

        if (seq.isEmpty() || labels.isEmpty())
            return 0;


        for (int i = 0; i < seq.size(); i++) {
            nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
            dm(i, seq, (int) nextRandom.get() % window, nextRandom, learningRate, labels, false, null);
        }

        return 0;
    }

    public void dm(int i, Sequence<T> sequence, int b, AtomicLong nextRandom, double alpha, List<T> labels,
                    boolean isInference, INDArray inferenceVector) {
        int end = window * 2 + 1 - b;

        T currentWord = sequence.getElementByIndex(i);

        List<Integer> intsList = new ArrayList<>();
        for (int a = b; a < end; a++) {
            if (a != window) {
                int c = i - window + a;
                if (c >= 0 && c < sequence.size()) {
                    T lastWord = sequence.getElementByIndex(c);

                    intsList.add(lastWord.getIndex());
                }
            }
        }

        // appending labels indexes
        if (labels != null)
            for (T label : labels) {
                intsList.add(label.getIndex());
            }

        int[] windowWords = new int[intsList.size()];
        for (int x = 0; x < windowWords.length; x++) {
            windowWords[x] = intsList.get(x);
        }

        // pass for underlying
        cbow.iterateSample(currentWord, windowWords, nextRandom, alpha, isInference, labels == null ? 0 : labels.size(),
                        configuration.isTrainElementsVectors(), inferenceVector);

        if (cbow.getBatch() != null && cbow.getBatch().size() >= configuration.getBatchSize()) {
            Nd4j.getExecutioner().exec(cbow.getBatch());
            cbow.getBatch().clear();
        }
    }

    @Override
    public boolean isEarlyTerminationHit() {
        return false;
    }

    /**
     * This method does training on previously unseen paragraph, and returns inferred vector
     *
     * @param sequence
     * @param nr
     * @param learningRate
     * @return
     */
    @Override
    public INDArray inferSequence(Sequence<T> sequence, long nr, double learningRate, double minLearningRate,
                    int iterations) {
        AtomicLong nextRandom = new AtomicLong(nr);

        // we probably don't want subsampling here
        // Sequence<T> seq = cbow.applySubsampling(sequence, nextRandom);
        // if (sequence.getSequenceLabel() == null) throw new IllegalStateException("Label is NULL");

        if (sequence.isEmpty())
            return null;

        Random random = Nd4j.getRandomFactory().getNewRandomInstance(configuration.getSeed() * sequence.hashCode(),
                        lookupTable.layerSize() + 1);
        INDArray ret = Nd4j.rand(new int[] {1, lookupTable.layerSize()}, random).subi(0.5)
                        .divi(lookupTable.layerSize());

        for (int iter = 0; iter < iterations; iter++) {
            for (int i = 0; i < sequence.size(); i++) {
                nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
                dm(i, sequence, (int) nextRandom.get() % window, nextRandom, learningRate, null, true, ret);
            }
            learningRate = ((learningRate - minLearningRate) / (iterations - iter)) + minLearningRate;
        }

        finish();

        return ret;
    }


    @Override
    public void finish() {
        if (cbow != null && cbow.getBatch() != null && !cbow.getBatch().isEmpty()) {
            Nd4j.getExecutioner().exec(cbow.getBatch());
            cbow.getBatch().clear();
        }
    }
}
