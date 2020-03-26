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

package org.deeplearning4j.spark.parameterserver.iterators;

import lombok.NonNull;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.ParallelMultiDataSetIterator;

import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This MultiDataSetIterator implementation does accumulation of MultiDataSets from different Spark executors, wrt Thread/Device Affinity
 *
 * @author raver119@gmail.com
 */
public class VirtualMultiDataSetIterator implements ParallelMultiDataSetIterator {

    protected final List<Iterator<MultiDataSet>> iterators;
    protected final AtomicInteger position;

    public VirtualMultiDataSetIterator(@NonNull List<Iterator<MultiDataSet>> iterators) {
        this.iterators = iterators;
        this.position = new AtomicInteger(0);
    }

    @Override
    public MultiDataSet next(int num) {
        return next();
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {

    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean hasNext() {
        // just checking if that's not the last iterator, or if that's the last one - check if it has something
        boolean ret = position.get() < iterators.size() - 1
                || (position.get() < iterators.size() && iterators.get(position.get()).hasNext());
        return ret;
    }

    @Override
    public MultiDataSet next() {
        // TODO: this solution isn't ideal, it assumes non-empty iterators all the time. Would be nice to do something here
        if (!iterators.get(position.get()).hasNext())
            position.getAndIncrement();

        return iterators.get(position.get()).next();
    }

    @Override
    public void remove() {
        // no-op
    }

    @Override
    public void attachThread(int producer) {

    }

    @Override
    public boolean hasNextFor() {
        return false;
    }

    @Override
    public boolean hasNextFor(int consumer) {
        return false;
    }

    @Override
    public MultiDataSet nextFor(int consumer) {
        return null;
    }

    @Override
    public MultiDataSet nextFor() {
        return null;
    }
}
