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

package org.deeplearning4j.datasets.iterator.parallel;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.ParallelDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.enums.InequalityHandling;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public abstract class BaseParallelDataSetIterator implements ParallelDataSetIterator {
    protected AtomicLong counter = new AtomicLong(0);

    protected InequalityHandling inequalityHandling;
    protected int numProducers;

    protected AtomicBoolean allDepleted = new AtomicBoolean(false);
    protected MultiBoolean states;
    protected MultiBoolean resetTracker;

    protected ThreadLocal<Integer> producerAffinity = new ThreadLocal<>();


    protected BaseParallelDataSetIterator(int numProducers) {
        states = new MultiBoolean(numProducers, true);
        resetTracker = new MultiBoolean(numProducers, false, true);
        this.numProducers = numProducers;
    }


    public boolean hasNext() {
        // if all producers are depleted - there's nothing to do here then
        if (states.allFalse() || allDepleted.get())
            return false;

        int curIdx = getCurrentProducerIndex();

        boolean hasNext = hasNextFor(curIdx);

        if (hasNext)
            return true;
        else
            states.set(hasNext, curIdx);

        if (states.allFalse())
            return false;

        switch (inequalityHandling) {
            // FIXME: RESET should be applicable ONLY to producers which return TRUE for resetSupported();
            case RESET: {
                resetTracker.set(true, curIdx);

                // we don't want to have endless loop here, so we only do reset until all producers depleted at least once
                if (resetTracker.allTrue()) {
                    allDepleted.set(true);
                    return false;
                }

                reset(curIdx);

                // triggering possible adsi underneath
                hasNextFor(curIdx);

                return true;
            }
            case RELOCATE: {
                // TODO: transparent switch to next producer should happen here
                while (!hasNext) {
                    stepForward();
                    hasNext = hasNextFor(getCurrentProducerIndex());
                    states.set(hasNext, getCurrentProducerIndex());

                    if (states.allFalse())
                        return false;
                }

                return true;
            }
            case PASS_NULL: {
                // we just return true here, no matter what's up
                return true;
            }
            case STOP_EVERYONE: {
                if (!states.allTrue())
                    return false;

                return true;
            }
            default:
                throw new ND4JIllegalStateException(
                                "Unknown InequalityHanding option was passed in: " + inequalityHandling);
        }
    }

    public DataSet next() {
        DataSet ds = nextFor(getCurrentProducerIndex());
        stepForward();
        return ds;
    }

    protected int getCurrentProducerIndex() {
        return (int) (counter.get() % numProducers);
    }

    protected void stepForward() {
        counter.getAndIncrement();
    }

    @Override
    public void reset() {
        for (int i = 0; i < numProducers; i++) {
            reset(i);
            states.set(true, i);
            resetTracker.set(false, i);
        }

        allDepleted.set(false);
    }

    @Override
    public void attachThread(int producer) {
        producerAffinity.set(producer);
    }

    @Override
    public boolean hasNextFor() {
        if (producerAffinity.get() == null)
            throw new ND4JIllegalStateException("attachThread(int) should be called prior to this call");

        return hasNextFor(producerAffinity.get());
    }

    @Override
    public DataSet nextFor() {
        if (producerAffinity.get() == null)
            throw new ND4JIllegalStateException("attachThread(int) should be called prior to this call");

        return nextFor(producerAffinity.get());
    }

    public abstract boolean hasNextFor(int consumer);

    public abstract DataSet nextFor(int consumer);

    protected abstract void reset(int consumer);

    @Override
    public int totalOutcomes() {
        return 0;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public int batch() {
        return 0;
    }

    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public void remove() {
        // no-op
    }
}
