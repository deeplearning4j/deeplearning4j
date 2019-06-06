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

package org.deeplearning4j.iterator.provider;

import lombok.NonNull;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;

/**
 * Simple class for conversion between LabelAwareIterator -> LabeledSentenceProvider for neural nets.
 * Since we already have converters for all other classes - this single converter allows us to accept all possible iterators
 *
 * @author raver119@gmail.com
 */
public class LabelAwareConverter implements LabeledSentenceProvider {
    private LabelAwareIterator backingIterator;
    private List<String> labels;

    public LabelAwareConverter(@NonNull LabelAwareIterator iterator, @NonNull List<String> labels) {
        this.backingIterator = iterator;
        this.labels = labels;
    }

    @Override
    public boolean hasNext() {
        return backingIterator.hasNext();
    }

    @Override
    public Pair<String, String> nextSentence() {
        LabelledDocument document = backingIterator.nextDocument();

        // TODO: probably worth to allow more then one label? i.e. pass same document twice, sequentially
        return Pair.makePair(document.getContent(), document.getLabels().get(0));
    }

    @Override
    public void reset() {
        backingIterator.reset();
    }

    @Override
    public int totalNumSentences() {
        return -1;
    }

    @Override
    public List<String> allLabels() {
        return labels;
    }

    @Override
    public int numLabelClasses() {
        return labels.size();
    }
}
