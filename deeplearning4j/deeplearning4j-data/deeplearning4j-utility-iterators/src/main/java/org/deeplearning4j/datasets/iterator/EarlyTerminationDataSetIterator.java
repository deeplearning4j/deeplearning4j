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

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;

/**
 * Builds an iterator that terminates once the number of minibatches returned with .next() is equal to a specified number
 * Note that a call to .next(num) is counted as a call to return a minibatch regardless of the value of num
 * This essentially restricts the data to this specified number of minibatches.
 */
public class EarlyTerminationDataSetIterator implements DataSetIterator {

    private DataSetIterator underlyingIterator;
    private int terminationPoint;
    private int minibatchCount = 0;

    /**
     * Constructor takes the iterator to wrap and the number of minibatches after which the call to hasNext()
     * will return false
     * @param underlyingIterator, iterator to wrap
     * @param terminationPoint, minibatches after which hasNext() will return false
     */
    public EarlyTerminationDataSetIterator(DataSetIterator underlyingIterator, int terminationPoint) {
        if (terminationPoint <= 0)
            throw new IllegalArgumentException(
                            "Termination point (the number of calls to .next() or .next(num)) has to be > 0");
        this.underlyingIterator = underlyingIterator;
        this.terminationPoint = terminationPoint;
    }

    @Override
    public DataSet next(int num) {
        if (minibatchCount < terminationPoint) {
            minibatchCount++;
            return underlyingIterator.next(num);
        } else {
            throw new RuntimeException("Calls to next have exceeded termination point.");
        }
    }

    @Override
    public int inputColumns() {
        return underlyingIterator.inputColumns();
    }

    @Override
    public int totalOutcomes() {
        return underlyingIterator.totalOutcomes();
    }

    @Override
    public boolean resetSupported() {
        return underlyingIterator.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return underlyingIterator.asyncSupported();
    }

    @Override
    public void reset() {
        minibatchCount = 0;
        underlyingIterator.reset();
    }

    @Override
    public int batch() {
        return underlyingIterator.batch();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        underlyingIterator.setPreProcessor(preProcessor);
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return underlyingIterator.getPreProcessor();
    }

    @Override
    public List<String> getLabels() {
        return underlyingIterator.getLabels();
    }

    @Override
    public boolean hasNext() {
        return underlyingIterator.hasNext() && minibatchCount < terminationPoint;
    }

    @Override
    public DataSet next() {
        if (minibatchCount < terminationPoint) {
            minibatchCount++;
            return underlyingIterator.next();
        } else {
            throw new RuntimeException("Calls to next have exceeded the allotted number of minibatches.");
        }
    }

    @Override
    public void remove() {
        underlyingIterator.remove();
    }
}
