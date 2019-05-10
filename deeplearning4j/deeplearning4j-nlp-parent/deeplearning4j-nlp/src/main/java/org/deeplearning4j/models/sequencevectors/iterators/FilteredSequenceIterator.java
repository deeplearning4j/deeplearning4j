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

package org.deeplearning4j.models.sequencevectors.iterators;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

/**
 * This implementation of SequenceIterator passes each sequence through specified vocabulary, filtering out SequenceElements that are not available in Vocabulary.
 * Please note: nextSequence() method can return empty sequence, if none of elements were found in attached vocabulary.
 *
 * @author raver119@gmail.com
 */
public class FilteredSequenceIterator<T extends SequenceElement> implements SequenceIterator<T> {

    private final SequenceIterator<T> underlyingIterator;
    private final VocabCache<T> vocabCache;

    /**
     * Creates Filtered SequenceIterator on top of another SequenceIterator and appropriate VocabCache instance
     *
     * @param iterator
     * @param vocabCache
     */
    public FilteredSequenceIterator(@NonNull SequenceIterator<T> iterator, @NonNull VocabCache<T> vocabCache) {
        this.vocabCache = vocabCache;
        this.underlyingIterator = iterator;
    }

    /**
     * Checks, if there's any more sequences left in underlying iterator
     * @return
     */
    @Override
    public boolean hasMoreSequences() {
        return underlyingIterator.hasMoreSequences();
    }

    /**
     * Returns filtered sequence, that contains sequence elements from vocabulary only.
     * Please note: it can return empty sequence, if no elements were found in vocabulary
     * @return
     */
    @Override
    public Sequence<T> nextSequence() {
        Sequence<T> originalSequence = underlyingIterator.nextSequence();
        Sequence<T> newSequence = new Sequence<>();

        if (originalSequence != null)
            for (T element : originalSequence.getElements()) {
                if (element != null && vocabCache.hasToken(element.getLabel())) {
                    newSequence.addElement(vocabCache.wordFor(element.getLabel()));
                }
            }

        newSequence.setSequenceId(originalSequence.getSequenceId());

        return newSequence;
    }

    /**
     * Resets iterator down to first sequence
     */
    @Override
    public void reset() {
        underlyingIterator.reset();
    }
}
