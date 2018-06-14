/*-
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.nd4j.linalg.dataset.api.iterator;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by susaneraly on 5/26/16.
 */
public class TestDataSetIterator implements DataSetIterator {

    private static final long serialVersionUID = -7569201667767185411L;
    private int curr = 0;
    private int batch = 10;
    private List<DataSet> list;
    private DataSetPreProcessor preProcessor;

    public TestDataSetIterator(DataSet dataset, int batch) {
        this(dataset.asList(), batch);
    }

    public TestDataSetIterator(List<DataSet> coll, int batch) {
        list = new ArrayList<>(coll);
        this.batch = batch;
    }

    /**
     * This makes an iterator from the given dataset and batchsize
     * ONLY for use in tests in nd4j
     * Initializes with a default batch of 5
     *
     * @param dataset the dataset to make the iterator from
     * @param batch   the batchsize for the iterator
     */
    public TestDataSetIterator(DataSet dataset) {
        this(dataset, 5);

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
        // FIXME: int cast
        return (int)list.get(0).getFeatureMatrix().columns();
    }

    @Override
    public int totalOutcomes() {
        // FIXME: int cast
        return (int) list.get(0).getLabels().columns();
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
    public synchronized void reset() {
        curr = 0;
    }

    @Override
    public int batch() {
        return batch;
    }

    @Override
    public void setPreProcessor(org.nd4j.linalg.dataset.api.DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return preProcessor;
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
        if (preProcessor != null)
            preProcessor.preProcess(d);
        return d;
    }

}
