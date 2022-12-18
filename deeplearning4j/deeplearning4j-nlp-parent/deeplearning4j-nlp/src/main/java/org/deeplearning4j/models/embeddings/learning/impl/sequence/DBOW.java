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

package org.deeplearning4j.models.embeddings.learning.impl.sequence;

import lombok.NonNull;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.learning.SequenceLearningAlgorithm;
import org.deeplearning4j.models.embeddings.learning.impl.elements.BatchItem;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

public class DBOW<T extends SequenceElement> implements SequenceLearningAlgorithm<T> {
    protected VocabCache<T> vocabCache;
    protected WeightLookupTable<T> lookupTable;
    protected VectorsConfiguration configuration;


    protected int window;
    protected boolean useAdaGrad;
    protected double negative;

    protected SkipGram<T> skipGram = new SkipGram<>();

    private static final Logger log = LoggerFactory.getLogger(DBOW.class);

    @Override
    public ElementsLearningAlgorithm<T> getElementsLearningAlgorithm() {
        return skipGram;
    }

    public DBOW() {

    }

    @Override
    public String getCodeName() {
        return "PV-DBOW";
    }

    @Override
    public void configure(@NonNull VocabCache<T> vocabCache, @NonNull WeightLookupTable<T> lookupTable,
                          @NonNull VectorsConfiguration configuration) {
        this.vocabCache = vocabCache;
        this.lookupTable = lookupTable;

        this.window = configuration.getWindow();
        this.useAdaGrad = configuration.isUseAdaGrad();
        this.negative = configuration.getNegative();
        this.configuration = configuration;

        skipGram.configure(vocabCache, lookupTable, configuration);
    }

    /**
     * DBOW doesn't involve any pretraining
     *
     * @param iterator
     */
    @Override
    public void pretrain(SequenceIterator<T> iterator) {

    }



    @Override
    public double learnSequence(@NonNull Sequence<T> sequence, @NonNull AtomicLong nextRandom, double learningRate) {
        dbow( sequence,  nextRandom, learningRate);
        return 0;
    }

    /**
     * DBOW has no reasons for early termination
     * @return
     */
    @Override
    public boolean isEarlyTerminationHit() {
        return false;
    }





    protected void dbow(Sequence<T> sequence, AtomicLong nextRandom, double alpha) {
        dbow(sequence,nextRandom,alpha,null);

    }

    protected void dbow(Sequence<T> sequence, AtomicLong nextRandom, double alpha,INDArray inferenceVector) {
        List<T> sentence = skipGram.applySubsampling(sequence, nextRandom).getElements();


        if (sequence.getSequenceLabel() == null)
            return;

        List<T> labels = new ArrayList<>();
        labels.addAll(sequence.getSequenceLabels());

        if (sentence.isEmpty() || labels.isEmpty())
            return;



        if (sequence.getSequenceLabel() == null)
            return;



        if (sentence.isEmpty() || labels.isEmpty())
            return;

        List<BatchItem<T>> batches = new ArrayList<>();
        for (T lastWord : labels) {
            for (T word : sentence) {
                if (word == null)
                    continue;

                BatchItem<T> batchItem = new BatchItem<>(word,lastWord,nextRandom.get(),alpha);
                batches.add(batchItem);


            }
        }


        if(inferenceVector != null)
            skipGram.doExec(batches,inferenceVector,configuration.isUseHierarchicSoftmax(),true);



        if(inferenceVector == null) {
            if(skipGram != null)
                skipGram.getBatch().addAll(batches);
            if(skipGram.getBatch().size() >= configuration.getBatchSize())
                finish();
        }
    }

    /**
     * This method does training on previously unseen paragraph, and returns inferred vector
     *
     * @param sequence
     * @param nextRandom
     * @param learningRate
     * @return
     */
    @Override
    public INDArray inferSequence(INDArray inferenceVector, Sequence<T> sequence, long nextRandom, double learningRate, double minLearningRate,
                                  int iterations) {
        AtomicLong nr = new AtomicLong(nextRandom);
        if (sequence.isEmpty())
            return null;



        INDArray ret = inferenceVector;

        for(int i = 0; i < iterations; i++) {
            dbow(sequence, nr, learningRate, ret);
            learningRate = ((learningRate - minLearningRate) / (iterations - i)) + minLearningRate;
        }

        return ret;
    }


    /**
     * This method does training on previously unseen paragraph, and returns inferred vector
     *
     * @param sequence
     * @param nextRandom
     * @param learningRate
     * @return
     */
    @Override
    public INDArray inferSequence(Sequence<T> sequence, long nextRandom, double learningRate, double minLearningRate,
                                  int iterations) {
        AtomicLong nr = new AtomicLong(nextRandom);
        if (sequence.isEmpty())
            return null;




        Random random = Nd4j.getRandomFactory().getNewRandomInstance(configuration.getSeed() * sequence.hashCode(),
                lookupTable.layerSize() + 1);
        INDArray ret = Nd4j.rand(random,new long[] {lookupTable.layerSize()}).subi(0.5)
                .divi(lookupTable.layerSize());

        return inferSequence(ret,sequence,nextRandom,learningRate,minLearningRate,iterations);
    }

    @Override
    public void finish() {
        if (skipGram != null && skipGram.getBatch() != null && !skipGram.getBatch().isEmpty()) {
            skipGram.finish();
        }
    }

    @Override
    public void finish(INDArray inferenceVector) {
        if (skipGram != null && skipGram.getBatch() != null && !skipGram.getBatch().isEmpty()) {
            skipGram.finish(inferenceVector);
        }
    }
}
