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

import lombok.Getter;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * A {@link MultiDataSetIterator} backed by an in-memory {@link List} of {@link MultiDataSet} instances.
 * Each call to {@link #next()} returns the next element in the list (one {@link MultiDataSet} at a time).
 * {@link #next(int)} merges up to {@code num} consecutive elements and returns the merged result.
 * <p>
 * Modeled after {@code org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator}.
 */
public class SimpleListMultiDataSetIterator implements MultiDataSetIterator {

    private static final long serialVersionUID = 1L;

    private int curr = 0;
    private final List<MultiDataSet> list;

    @Getter
    private MultiDataSetPreProcessor preProcessor;

    /**
     * @param coll Collection of {@link MultiDataSet} instances to iterate over
     */
    public SimpleListMultiDataSetIterator(Collection<? extends MultiDataSet> coll) {
        this.list = new ArrayList<>(coll);
    }

    @Override
    public synchronized boolean hasNext() {
        return curr < list.size();
    }

    @Override
    public synchronized MultiDataSet next() {
        return next(1);
    }

    @Override
    public synchronized MultiDataSet next(int num) {
        int end = Math.min(curr + num, list.size());
        List<MultiDataSet> batch = new ArrayList<>(end - curr);
        for (; curr < end; curr++) {
            batch.add(list.get(curr));
        }
        MultiDataSet result = org.nd4j.linalg.dataset.MultiDataSet.merge(batch);
        if (preProcessor != null) {
            preProcessor.preProcess(result);
        }
        return result;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        // Already in memory — no benefit to asynchronous prefetching
        return false;
    }

    @Override
    public synchronized void reset() {
        curr = 0;
    }
}
