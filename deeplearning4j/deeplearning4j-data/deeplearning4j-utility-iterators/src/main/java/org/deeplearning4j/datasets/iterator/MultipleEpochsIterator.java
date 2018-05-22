/*-
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
import com.google.common.collect.Lists;
import lombok.Getter;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.NoSuchElementException;
import java.util.concurrent.atomic.AtomicLong;


/**
 * A dataset iterator for doing multiple passes over a dataset
 *
 * @deprecated Does not properly trigger the incrementing of epoch counts in MultiLayerNetwork/ComputationGraph.
 * Use MultiLayerNetwork/ComputationGraph.fit(DataSetIterator, int numEpochs) instead
 */
@Deprecated
public class MultipleEpochsIterator implements DataSetIterator {
    @VisibleForTesting
    protected int epochs = 0;
    protected int numEpochs;
    protected int batch = 0;
    protected int lastBatch = batch;
    protected DataSetIterator iter;
    protected DataSet ds;
    protected List<DataSet> batchedDS = Lists.newArrayList();
    protected static final Logger log = LoggerFactory.getLogger(MultipleEpochsIterator.class);
    @Getter
    protected DataSetPreProcessor preProcessor;
    protected boolean newEpoch = false;
    protected AtomicLong iterationsCounter = new AtomicLong(0);
    protected long totalIterations = Long.MAX_VALUE;

    @Deprecated
    public MultipleEpochsIterator(int numEpochs, DataSetIterator iter) {
        this.numEpochs = numEpochs;
        this.iter = iter;
    }

    @Deprecated
    public MultipleEpochsIterator(int numEpochs, DataSetIterator iter, int queueSize) {
        this.numEpochs = numEpochs;
        this.iter = iter;
    }

    @Deprecated
    public MultipleEpochsIterator(DataSetIterator iter, int queueSize, long totalIterations) {
        this.numEpochs = Integer.MAX_VALUE;
        this.iter = iter;
        this.totalIterations = totalIterations;
    }

    @Deprecated
    public MultipleEpochsIterator(DataSetIterator iter, long totalIterations) {
        this.numEpochs = Integer.MAX_VALUE;
        this.iter = iter;
        this.totalIterations = totalIterations;
    }

    @Deprecated
    public MultipleEpochsIterator(int numEpochs, DataSet ds) {
        this.numEpochs = numEpochs;
        this.ds = ds;
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
        if (!hasNext()) {
            throw new NoSuchElementException("No next element");
        }
        DataSet next;
        batch++;
        iterationsCounter.incrementAndGet();
        if (iter == null) {
            // return full DataSet
            if (num == -1) {
                next = ds;
                if (epochs < numEpochs)
                    trackEpochs();
            }
            // return DataSet broken into batches
            else {
                if (batchedDS.isEmpty() && num > 0)
                    batchedDS = ds.batchBy(num);
                next = batchedDS.get(batch);
                if (batch + 1 == batchedDS.size()) {
                    trackEpochs();
                    if (epochs < numEpochs)
                        batch = -1;
                }
            }
        } else {
            next = (num == -1 ? iter.next() : iter.next(num));
            if (next == null) {
                throw new IllegalStateException("Iterator returned null DataSet");
            }
            if (!iter.hasNext()) {
                trackEpochs();
                // track number of epochs and won't reset if it's over
                if (epochs < numEpochs) {
                    iter.reset();
                    lastBatch = batch;
                    batch = 0;
                }
            }
        }
        if (preProcessor != null)
            preProcessor.preProcess(next);
        return next;
    }

    public void trackEpochs() {
        epochs++;
        newEpoch = true;
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

    @Override
    public boolean resetSupported() {
        return iter.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return iter.asyncSupported();
    }

    /**
     * Resets the iterator back to the beginning
     */
    @Override
    public void reset() {
        if (!iter.resetSupported()) {
            throw new IllegalStateException(
                            "Cannot reset MultipleEpochsIterator with base iter that does not support reset");
        }
        epochs = 0;
        lastBatch = batch;
        batch = 0;
        iterationsCounter.set(0);
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
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
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
        if (iterationsCounter.get() >= totalIterations)
            return false;

        if (newEpoch) {
            log.info("Epoch " + epochs + ", number of batches completed " + lastBatch);
            newEpoch = false;
        }
        if (iter == null)
            return (epochs < numEpochs) && ((!batchedDS.isEmpty() && batchedDS.size() > batch) || batchedDS.isEmpty());
        else
            // either there are still epochs to complete or its the first epoch
            return (epochs < numEpochs) || (iter.hasNext() && (epochs == 0 || epochs == numEpochs));
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
