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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Wraps a data applyTransformToDestination collection
 *
 * @author Adam Gibson
 */
public class ListDataSetIterator<T extends DataSet> implements DataSetIterator {

    private static final long serialVersionUID = -7569201667767185411L;
    private int curr = 0;
    private int batch = 10;
    private List<T> list;
    @Getter
    private DataSetPreProcessor preProcessor;

    public ListDataSetIterator(Collection<T> coll, int batch) {
        list = new ArrayList<>(coll);
        this.batch = batch;

    }

    /**
     * Initializes with a batch of 5
     *
     * @param coll the collection to iterate over
     */
    public ListDataSetIterator(Collection<T> coll) {
        this(coll, 5);

    }

    @Override
    public synchronized boolean hasNext() {
        return curr < list.size();
    }

    @Override
    public synchronized DataSet next() {
        return next(batch);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        return list.get(0).getFeatures().columns();
    }

    @Override
    public int totalOutcomes() {
        return list.get(0).getLabels().columns();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        //Already in memory -> doesn't make sense to prefetch
        return false;
    }

    @Override
    public synchronized void reset() {
        curr = 0;
    }

    @Override
    public int batch() {
        return batch;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }


    @Override
    public DataSet next(int num) {
        int end = curr + num;

        List<DataSet> r = new ArrayList<>();
        if (end >= list.size())
            end = list.size();
        for (; curr < end; curr++) {
            r.add(list.get(curr));
        }

        DataSet d = DataSet.merge(r);
        if (preProcessor != null) {
            if (!d.isPreProcessed()) {
                preProcessor.preProcess(d);
                d.markAsPreProcessed();
            }
        }
        return d;
    }


}
