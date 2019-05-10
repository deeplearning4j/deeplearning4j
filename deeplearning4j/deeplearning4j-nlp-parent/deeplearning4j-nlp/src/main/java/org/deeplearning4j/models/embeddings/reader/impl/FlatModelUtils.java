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

package org.deeplearning4j.models.embeddings.reader.impl;

import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Counter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;

/**
 * This model reader is suited for model tests, and for cases where flat scan against elements is required.
 *
 * PLEASE NOTE: This reader does NOT normalize underlying weights, it stays intact
 *
 * @author raver119@gmail.com
 */
public class FlatModelUtils<T extends SequenceElement> extends BasicModelUtils<T> {
    private static final Logger log = LoggerFactory.getLogger(FlatModelUtils.class);

    public FlatModelUtils() {

    }

    /**
     * This method does full scan against whole vocabulary, building descending list of similar words
     * @param label
     * @param n
     * @return
     */
    @Override
    public Collection<String> wordsNearest(String label, int n) {
        Collection<String> collection = wordsNearest(lookupTable.vector(label), n);
        if (collection.contains(label))
            collection.remove(label);
        return collection;
    }

    /**
     * This method does full scan against whole vocabulary, building descending list of similar words
     *
     * @param words
     * @param top
     * @return the words nearest the mean of the words
     */
    @Override
    public Collection<String> wordsNearest(INDArray words, int top) {
        Counter<String> distances = new Counter<>();

        for (String s : vocabCache.words()) {
            INDArray otherVec = lookupTable.vector(s);
            double sim = Transforms.cosineSim(Transforms.unitVec(words.dup()), Transforms.unitVec(otherVec.dup()));
            distances.incrementCount(s, (float) sim);
        }

        distances.keepTopNElements(top);
        return distances.keySetSorted();
    }
}
