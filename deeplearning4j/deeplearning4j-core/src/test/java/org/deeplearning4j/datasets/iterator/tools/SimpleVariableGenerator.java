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

package org.deeplearning4j.datasets.iterator.tools;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author raver119@gmail.com
 */
public class SimpleVariableGenerator implements DataSetIterator {
    private long seed;
    private int numBatches;
    private int batchSize;
    private int numFeatures;
    private int numLabels;

    private AtomicInteger counter = new AtomicInteger(0);

    public SimpleVariableGenerator(long seed, int numBatches, int batchSize, int numFeatures, int numLabels) {
        this.seed = seed;
        this.numBatches = numBatches;
        this.batchSize = batchSize;
        this.numFeatures = numFeatures;
        this.numLabels = numLabels;
    }

    @Override
    public DataSet next() {
        INDArray features = Nd4j.create(batchSize, numFeatures).assign(counter.get());
        INDArray labels = Nd4j.create(batchSize, numFeatures).assign(counter.getAndIncrement() + 0.5);
        return new DataSet(features, labels);
    }

    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        return numFeatures;
    }

    @Override
    public int totalOutcomes() {
        return numLabels;
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
        counter.set(0);
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return counter.get() < numBatches;
    }

    @Override
    public void remove() {

    }
}
