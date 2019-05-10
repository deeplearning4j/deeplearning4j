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

package org.deeplearning4j.spark.models.sequencevectors.learning.sequence;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.models.sequencevectors.learning.SparkSequenceLearningAlgorithm;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseSparkSequenceLearningAlgorithm implements SparkSequenceLearningAlgorithm {
    protected transient VocabCache<ShallowSequenceElement> vocabCache;
    protected transient VectorsConfiguration vectorsConfiguration;
    protected transient ElementsLearningAlgorithm<ShallowSequenceElement> elementsLearningAlgorithm;

    @Override
    public void configure(VocabCache<ShallowSequenceElement> vocabCache,
                    WeightLookupTable<ShallowSequenceElement> lookupTable, VectorsConfiguration configuration) {
        this.vocabCache = vocabCache;
        this.vectorsConfiguration = configuration;
    }

    @Override
    public void pretrain(SequenceIterator<ShallowSequenceElement> iterator) {
        // no-op by default
    }

    @Override
    public boolean isEarlyTerminationHit() {
        return false;
    }

    @Override
    public INDArray inferSequence(Sequence<ShallowSequenceElement> sequence, long nextRandom, double learningRate,
                    double minLearningRate, int iterations) {
        throw new UnsupportedOperationException();
    }

    @Override
    public ElementsLearningAlgorithm<ShallowSequenceElement> getElementsLearningAlgorithm() {
        return elementsLearningAlgorithm;
    }

    @Override
    public void finish() {
        // no-op on spark
    }
}
