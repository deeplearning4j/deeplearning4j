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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;

/**
 * This wrapper takes your existing DataSetIterator implementation and prevents asynchronous prefetch
 * when using methods such as {@code MultiLayerNetwork.fit(DataSetIterator)}
 * This is mainly used for debugging purposes; generally an iterator that isn't safe to asynchronously prefetch from
 * should simply return {@code asyncSupported() == false}
 *
 * @author raver119@gmail.com
 */
public class AsyncShieldDataSetIterator implements DataSetIterator {
    private DataSetIterator backingIterator;

    /**
     * @param iterator Iterator to wrop, to disable asynchronous prefetching for
     */
    public AsyncShieldDataSetIterator(@NonNull DataSetIterator iterator) {
        this.backingIterator = iterator;
    }

    /**
     * Like the standard next method but allows a
     * customizable number of examples returned
     *
     * @param num the number of examples
     * @return the next data applyTransformToDestination
     */
    @Override
    public DataSet next(int num) {
        return backingIterator.next(num);
    }

    /**
     * Input columns for the dataset
     *
     * @return
     */
    @Override
    public int inputColumns() {
        return backingIterator.inputColumns();
    }

    /**
     * The number of labels for the dataset
     *
     * @return
     */
    @Override
    public int totalOutcomes() {
        return backingIterator.totalOutcomes();
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
     * Batch size
     *
     * @return
     */
    @Override
    public int batch() {
        return backingIterator.batch();
    }

    /**
     * Set a pre processor
     *
     * @param preProcessor a pre processor to set
     */
    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        backingIterator.setPreProcessor(preProcessor);
    }

    /**
     * Returns preprocessors, if defined
     *
     * @return
     */
    @Override
    public DataSetPreProcessor getPreProcessor() {
        return backingIterator.getPreProcessor();
    }

    /**
     * Get dataset iterator record reader labels
     */
    @Override
    public List<String> getLabels() {
        return backingIterator.getLabels();
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
    public DataSet next() {
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
