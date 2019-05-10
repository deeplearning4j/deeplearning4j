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

package org.deeplearning4j.spark.models.sequencevectors.learning.elements;

import lombok.NonNull;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.models.sequencevectors.learning.SparkElementsLearningAlgorithm;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseSparkLearningAlgorithm implements SparkElementsLearningAlgorithm {
    protected transient VocabCache<ShallowSequenceElement> vocabCache;
    protected transient VectorsConfiguration vectorsConfiguration;
    protected transient AtomicLong nextRandom;

    protected BaseSparkLearningAlgorithm() {

    }

    @Override
    public double learnSequence(Sequence<ShallowSequenceElement> sequence, AtomicLong nextRandom, double learningRate) {
        // no-op
        return 0;
    }

    @Override
    public void configure(VocabCache<ShallowSequenceElement> vocabCache,
                    WeightLookupTable<ShallowSequenceElement> lookupTable, VectorsConfiguration configuration) {
        this.vocabCache = vocabCache;
        this.vectorsConfiguration = configuration;
    }

    @Override
    public void pretrain(SequenceIterator<ShallowSequenceElement> iterator) {
        // no-op
    }

    @Override
    public boolean isEarlyTerminationHit() {
        return false;
    }

    @Override
    public void finish() {
        // no-op
    }

    public static Sequence<ShallowSequenceElement> applySubsampling(@NonNull Sequence<ShallowSequenceElement> sequence,
                    @NonNull AtomicLong nextRandom, long totalElementsCount, double prob) {
        Sequence<ShallowSequenceElement> result = new Sequence<>();

        // subsampling implementation, if subsampling threshold met, just continue to next element
        if (prob > 0) {
            result.setSequenceId(sequence.getSequenceId());
            if (sequence.getSequenceLabels() != null)
                result.setSequenceLabels(sequence.getSequenceLabels());
            if (sequence.getSequenceLabel() != null)
                result.setSequenceLabel(sequence.getSequenceLabel());

            for (ShallowSequenceElement element : sequence.getElements()) {
                double numWords = (double) totalElementsCount;
                double ran = (Math.sqrt(element.getElementFrequency() / (prob * numWords)) + 1) * (prob * numWords)
                                / element.getElementFrequency();

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
