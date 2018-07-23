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

import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;
import java.util.NoSuchElementException;

/**
 * A very simple adapter class for converting a single DataSet to a DataSetIterator.
 * Returns a single DataSet as-is, for each epoch
 *
 * @author Alex Black
 */
public class SingletonDataSetIterator implements DataSetIterator {

    private final DataSet dataSet;
    private boolean hasNext = true;
    private boolean preprocessed = false;
    @Getter @Setter
    private DataSetPreProcessor preProcessor;

    public SingletonDataSetIterator(DataSet multiDataSet) {
        this.dataSet = multiDataSet;
    }

    @Override
    public DataSet next(int num) {
        return next();
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
        return false;
    }

    @Override
    public void reset() {
        hasNext = true;
    }

    @Override
    public int batch() {
        return 0;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return hasNext;
    }

    @Override
    public DataSet next() {
        if (!hasNext) {
            throw new NoSuchElementException("No elements remaining");
        }
        hasNext = false;
        if (preProcessor != null && !preprocessed) {
            preProcessor.preProcess(dataSet);
            preprocessed = true;
        }
        return dataSet;
    }

    @Override
    public void remove() {
        //No op
    }
}
