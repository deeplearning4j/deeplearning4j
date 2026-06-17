/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.dataset;

import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.List;

/**
 * A simple in-memory {@link MultiDataSetIterator} backed by a {@link List} of
 * {@link MultiDataSet} instances.
 *
 * <p>Each call to {@link #next()} returns the next dataset in the list.
 * {@link #reset()} rewinds the cursor to the beginning.  This iterator is
 * primarily used for small in-memory training scenarios (e.g. SFT pipelines
 * that build datasets at runtime).</p>
 *
 * @see org.nd4j.autodiff.samediff.training.SFTTrainingPipeline
 */
public class SimpleListMultiDataSetIterator implements MultiDataSetIterator {

    private final List<MultiDataSet> datasets;
    private int cursor;
    private MultiDataSetPreProcessor preProcessor;

    /**
     * Create an iterator over the given list of datasets.
     *
     * @param datasets list of {@link MultiDataSet} instances (not null, not modified)
     */
    public SimpleListMultiDataSetIterator(List<MultiDataSet> datasets) {
        if (datasets == null) throw new IllegalArgumentException("datasets must not be null");
        this.datasets = datasets;
        this.cursor   = 0;
    }

    @Override
    public boolean hasNext() {
        return cursor < datasets.size();
    }

    @Override
    public MultiDataSet next() {
        return next(1);
    }

    @Override
    public MultiDataSet next(int num) {
        if (!hasNext()) throw new java.util.NoSuchElementException("No more datasets");

        int end = Math.min(cursor + num, datasets.size());

        MultiDataSet result;
        if (end - cursor == 1) {
            result = datasets.get(cursor);
        } else {
            List<MultiDataSet> batch = datasets.subList(cursor, end);
            result = org.nd4j.linalg.dataset.MultiDataSet.merge(batch);
        }

        cursor = end;

        if (preProcessor != null) {
            preProcessor.preProcess(result);
        }

        return result;
    }

    @Override
    public void reset() {
        cursor = 0;
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
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("remove() is not supported");
    }
}
