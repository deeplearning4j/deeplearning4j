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

package org.nd4j.linalg.dataset.api.iterator;

import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Ede Meijer
 */
public class TestMultiDataSetIterator implements MultiDataSetIterator {
    private int curr = 0;
    private int batch = 10;
    private List<MultiDataSet> list;
    private MultiDataSetPreProcessor preProcessor;

    /**
     * Makes an iterator from the given datasets. DataSets are expected to are batches of exactly 1 example.
     * ONLY for use in tests in nd4j
     */
    public TestMultiDataSetIterator(int batch, MultiDataSet... dataset) {
        list = Arrays.asList(dataset);
        this.batch = batch;
    }

    @Override
    public MultiDataSet next(int num) {
        int end = curr + num;

        List<MultiDataSet> r = new ArrayList<>();
        if (end >= list.size()) {
            end = list.size();
        }
        for (; curr < end; curr++) {
            r.add(list.get(curr));
        }

        MultiDataSet d = org.nd4j.linalg.dataset.MultiDataSet.merge(r);
        if (preProcessor != null) {
            preProcessor.preProcess(d);
        }
        return d;
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return this.preProcessor;
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
        curr = 0;
    }

    @Override
    public boolean hasNext() {
        return curr < list.size();
    }

    @Override
    public MultiDataSet next() {
        return next(batch);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }
}
