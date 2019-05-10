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

package org.deeplearning4j.datasets.iterator;

import lombok.NonNull;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * This wrapper takes your existing MultiDataSetIterator implementation and prevents asynchronous prefetch
 *
 * @author raver119@gmail.com
 */
public class AsyncShieldMultiDataSetIterator implements MultiDataSetIterator {
    private MultiDataSetIterator backingIterator;

    public AsyncShieldMultiDataSetIterator(@NonNull MultiDataSetIterator iterator) {
        this.backingIterator = iterator;
    }

    /**
     * Fetch the next 'num' examples. Similar to the next method, but returns a specified number of examples
     *
     * @param num Number of examples to fetch
     */
    @Override
    public MultiDataSet next(int num) {
        return backingIterator.next(num);
    }

    /**
     * Set the preprocessor to be applied to each MultiDataSet, before each MultiDataSet is returned.
     *
     * @param preProcessor MultiDataSetPreProcessor. May be null.
     */
    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        backingIterator.setPreProcessor(preProcessor);
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return backingIterator.getPreProcessor();
    }

    /**
     * Is resetting supported by this DataSetIterator? Many DataSetIterators do support resetting,
     * but some don't
     *
     * @return true if reset method is supported; false otherwise
     */
    @Override
    public boolean resetSupported() {
        return backingIterator.resetSupported();
    }

    /**
     /**
     * Does this DataSetIterator support asynchronous prefetching of multiple DataSet objects?
     *
     * PLEASE NOTE: This iterator ALWAYS returns FALSE
     *
     * @return true if asynchronous prefetching from this iterator is OK; false if asynchronous prefetching should not
     * be used with this iterator
     */
    @Override
    public boolean asyncSupported() {
        return false;
    }

    /**
     * Resets the iterator back to the beginning
     */
    @Override
    public void reset() {
        backingIterator.reset();
    }

    /**
     * Returns {@code true} if the iteration has more elements.
     * (In other words, returns {@code true} if {@link #next} would
     * return an element rather than throwing an exception.)
     *
     * @return {@code true} if the iteration has more elements
     */
    @Override
    public boolean hasNext() {
        return backingIterator.hasNext();
    }

    /**
     * Returns the next element in the iteration.
     *
     * @return the next element in the iteration
     */
    @Override
    public MultiDataSet next() {
        return backingIterator.next();
    }

    /**
     * Removes from the underlying collection the last element returned
     * by this iterator (optional operation).  This method can be called
     * only once per call to {@link #next}.  The behavior of an iterator
     * is unspecified if the underlying collection is modified while the
     * iteration is in progress in any way other than by calling this
     * method.
     *
     * @throws UnsupportedOperationException if the {@code remove}
     *                                       operation is not supported by this iterator
     * @throws IllegalStateException         if the {@code next} method has not
     *                                       yet been called, or the {@code remove} method has already
     *                                       been called after the last call to the {@code next}
     *                                       method
     * @implSpec The default implementation throws an instance of
     * {@link UnsupportedOperationException} and performs no other action.
     */
    @Override
    public void remove() {
        // no-op
    }
}
