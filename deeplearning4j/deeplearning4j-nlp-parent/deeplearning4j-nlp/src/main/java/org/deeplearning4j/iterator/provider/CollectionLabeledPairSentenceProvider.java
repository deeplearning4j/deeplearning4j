/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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

package org.deeplearning4j.iterator.provider;

import lombok.NonNull;
import org.deeplearning4j.iterator.LabeledPairSentenceProvider;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.linalg.util.MathUtils;

import java.util.*;

/**
 * Iterate over a pair of sentences/documents,
 * where the sentences and labels are provided in lists.
 *
 */
public class CollectionLabeledPairSentenceProvider implements LabeledPairSentenceProvider {

    private final List<String> sentenceL;
    private final List<String> sentenceR;
    private final List<String> labels;
    private final Random rng;
    private final int[] order;
    private final List<String> allLabels;

    private int cursor = 0;

    /**
     * Lists containing sentences to iterate over with a third for labels
     * Sentences in the same position in the first two lists are considered a pair
     * @param sentenceL
     * @param sentenceR
     * @param labelsForSentences
     */
    public CollectionLabeledPairSentenceProvider(@NonNull List<String> sentenceL, @NonNull List<String> sentenceR,
                                                 @NonNull List<String> labelsForSentences) {
        this(sentenceL, sentenceR, labelsForSentences, new Random());
    }

    /**
     * Lists containing sentences to iterate over with a third for labels
     * Sentences in the same position in the first two lists are considered a pair
     * @param sentenceL
     * @param sentenceR
     * @param labelsForSentences
     * @param rng If null, list order is not shuffled
     */
    public CollectionLabeledPairSentenceProvider(@NonNull List<String> sentenceL, List<String> sentenceR, @NonNull List<String> labelsForSentences,
                                                 Random rng) {
        if (sentenceR.size() != sentenceL.size()) {
            throw new IllegalArgumentException("Sentence lists must be same size (first list size: "
                    + sentenceL.size() + ", second list size: " + sentenceR.size() + ")");
        }
        if (sentenceR.size() != labelsForSentences.size()) {
            throw new IllegalArgumentException("Sentence pairs and labels must be same size (sentence pair size: "
                    + sentenceR.size() + ", labels size: " + labelsForSentences.size() + ")");
        }

        this.sentenceL = sentenceL;
        this.sentenceR = sentenceR;
        this.labels = labelsForSentences;
        this.rng = rng;
        if (rng == null) {
            order = null;
        } else {
            order = new int[sentenceR.size()];
            for (int i = 0; i < sentenceR.size(); i++) {
                order[i] = i;
            }

            MathUtils.shuffleArray(order, rng);
        }

        //Collect set of unique labels for all sentences
        Set<String> uniqueLabels = new HashSet<>(labelsForSentences);
        allLabels = new ArrayList<>(uniqueLabels);
        Collections.sort(allLabels);
    }

    @Override
    public boolean hasNext() {
        return cursor < sentenceR.size();
    }

    @Override
    public Triple<String, String, String> nextSentencePair() {
        Preconditions.checkState(hasNext(),"No next element available");
        int idx;
        if (rng == null) {
            idx = cursor++;
        } else {
            idx = order[cursor++];
        }
        return new Triple<>(sentenceL.get(idx), sentenceR.get(idx), labels.get(idx));
    }

    @Override
    public void reset() {
        cursor = 0;
        if (rng != null) {
            MathUtils.shuffleArray(order, rng);
        }
    }

    @Override
    public int totalNumSentences() {
        return sentenceR.size();
    }

    @Override
    public List<String> allLabels() {
        return allLabels;
    }

    @Override
    public int numLabelClasses() {
        return allLabels.size();
    }
}

