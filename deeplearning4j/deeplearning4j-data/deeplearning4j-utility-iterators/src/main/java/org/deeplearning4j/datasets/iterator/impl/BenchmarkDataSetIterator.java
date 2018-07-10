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

package org.deeplearning4j.datasets.iterator.impl;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Dataset iterator for simulated inputs, or input derived from a DataSet example. Primarily
 * used for benchmarking.
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public class BenchmarkDataSetIterator implements DataSetIterator {
    private INDArray baseFeatures;
    private INDArray baseLabels;
    private long limit;
    private AtomicLong counter = new AtomicLong(0);

    public BenchmarkDataSetIterator(int[] featuresShape, int numLabels, int totalIterations) {
        this(featuresShape, numLabels, totalIterations, -1, -1);
    }

    public BenchmarkDataSetIterator(int[] featuresShape, int numLabels, int totalIterations, int gridWidth, int gridHeight) {
        this.baseFeatures = Nd4j.rand(featuresShape);
        this.baseLabels = gridWidth > 0 && gridHeight > 0
                        ? Nd4j.create(featuresShape[0], numLabels, gridWidth, gridHeight)
                        : Nd4j.create(featuresShape[0], numLabels);
        this.baseLabels.getColumn(1).assign(1.0);

        Nd4j.getExecutioner().commit();
        this.limit = totalIterations;
    }

    public BenchmarkDataSetIterator(DataSet example, int totalIterations) {
        this.baseFeatures = example.getFeatures().dup();
        this.baseLabels = example.getLabels().dup();

        Nd4j.getExecutioner().commit();
        this.limit = totalIterations;
    }

    @Override
    public DataSet next(int i) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        return 0;
    }

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
        return true;
    }

    @Override
    public void reset() {
        this.counter.set(0);
    }

    @Override
    public int batch() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return null;
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
        return counter.get() < limit;
    }

    /**
     * Returns the next element in the iteration.
     *
     * @return the next element in the iteration
     */
    @Override
    public DataSet next() {
        counter.incrementAndGet();

        DataSet ds = new DataSet(baseFeatures, baseLabels);

        return ds;
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

    }
}
