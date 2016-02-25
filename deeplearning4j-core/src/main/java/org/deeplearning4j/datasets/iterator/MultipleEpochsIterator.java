/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.datasets.iterator;

import com.google.common.annotations.VisibleForTesting;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * A dataset iterator for doing multiple passes over a dataset
 */
public class MultipleEpochsIterator implements DataSetIterator {
    @VisibleForTesting
    protected int numPasses;
    protected int batch = 0;
    protected DataSetIterator iter;
    protected int passes = 0;
    protected static final Logger log = LoggerFactory.getLogger(MultipleEpochsIterator.class);
    protected DataSetPreProcessor preProcessor;

    public MultipleEpochsIterator(int numPasses,DataSetIterator iter) {
        this.numPasses = numPasses;
        this.iter = iter;
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
        if(!iter.hasNext()) {
            passes++;
            log.info("Epoch " + passes + ", number of batches completed " + batch);
            if(passes < numPasses) {
                batch = 0;
                iter.reset();
            } else {
                return null;
            }
        }
        batch++;

        DataSet next = num == -1? iter.next(): iter.next(num);
        if(preProcessor != null)
            preProcessor.preProcess(next);
        return next;
    }


    @Override
    public DataSet next() {
        return next(-1);
    }

    /**
     * Total examples in the iterator
     *
     * @return
     */
    @Override
    public int totalExamples() {
        return iter.totalExamples();
    }

    /**
     * Input columns for the dataset
     *
     * @return
     */
    @Override
    public int inputColumns() {
        return iter.inputColumns();
    }

    /**
     * The number of labels for the dataset
     *
     * @return
     */
    @Override
    public int totalOutcomes() {
        return iter.totalOutcomes();
    }

    /**
     * Resets the iterator back to the beginning
     */
    @Override
    public void reset() {
        passes = 0;
        batch = 0;
        iter.reset();
    }

    /**
     * Batch size
     *
     * @return
     */
    @Override
    public int batch() {
        return iter.batch();
    }

    /**
     * The current cursor if applicable
     *
     * @return
     */
    @Override
    public int cursor() {
        return iter.cursor();
    }

    /**
     * Total number of examples in the dataset
     *
     * @return
     */
    @Override
    public int numExamples() {
        return iter.numExamples();
    }

    @Override
    public void setPreProcessor(org.nd4j.linalg.dataset.api.DataSetPreProcessor preProcessor) {
        this.preProcessor = (DataSetPreProcessor) preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return iter.getLabels();
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
        return iter.hasNext() || passes < numPasses;
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
     */
    @Override
    public void remove() {
           iter.remove();
    }
}
