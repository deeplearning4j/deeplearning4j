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

import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.NoSuchElementException;

/**
 * A very simple adapter class for converting a single MultiDataSet to a MultiDataSetIterator.
 * Returns a single MultiDataSet as-is, once for each epoch
 *
 * @author Alex Black
 */
public class SingletonMultiDataSetIterator implements MultiDataSetIterator {

    private final MultiDataSet multiDataSet;
    private boolean hasNext = true;
    private boolean preprocessed = false;
    private MultiDataSetPreProcessor preProcessor;

    /**
     * @param multiDataSet The underlying MultiDataSet to return
     */
    public SingletonMultiDataSetIterator(MultiDataSet multiDataSet) {
        this.multiDataSet = multiDataSet;
    }

    @Override
    public MultiDataSet next(int num) {
        return next();
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return preProcessor;
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
    public boolean hasNext() {
        return hasNext;
    }

    @Override
    public MultiDataSet next() {
        if (!hasNext) {
            throw new NoSuchElementException("No elements remaining");
        }
        hasNext = false;
        if (preProcessor != null && !preprocessed) {
            preProcessor.preProcess(multiDataSet);
            preprocessed = true;
        }
        return multiDataSet;
    }

    @Override
    public void remove() {
        //No op
    }
}
