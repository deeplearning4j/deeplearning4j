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

/**
 * Synchronized version of AbstractSeuqenceIterator, implemented on top of it.
 * Suitable for cases with non-strict multithreading environment, since it's just synchronized wrapper
 *
 * @author raver119@gmail.com
 */
public class SynchronizedSequenceIterator<T extends SequenceElement> implements SequenceIterator<T> {
    protected SequenceIterator<T> underlyingIterator;

    /**
     * Creates SynchronizedSequenceIterator on top of any SequenceIterator
     * @param iterator
     */
    public SynchronizedSequenceIterator(@NonNull SequenceIterator<T> iterator) {
        this.underlyingIterator = iterator;
    }

    /**
     * Checks, if there's any more sequences left in data source
     * @return
     */
    @Override
    public synchronized boolean hasMoreSequences() {
        return underlyingIterator.hasMoreSequences();
    }

    /**
     * Returns next sequence from data source
     *
     * @return
     */
    @Override
    public synchronized Sequence<T> nextSequence() {
        return underlyingIterator.nextSequence();
    }

    /**
     * This method resets underlying iterator
     */
    @Override
    public synchronized void reset() {
        underlyingIterator.reset();
    }
}
