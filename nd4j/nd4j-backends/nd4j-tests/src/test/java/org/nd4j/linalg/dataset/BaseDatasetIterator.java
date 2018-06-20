/*-
 *
 *  * Copyright 2015 Skymind,Inc.
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

package org.nd4j.linalg.dataset;

import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher;

import java.util.List;

/**
 * Baseline implementation includes
 * control over the data fetcher and some basic
 * getters for metadata
 * @author Adam Gibson
 *
 */
public class BaseDatasetIterator implements DataSetIterator {


    private static final long serialVersionUID = -116636792426198949L;
    protected int batch, numExamples;
    protected DataSetFetcher fetcher;
    protected DataSetPreProcessor preProcessor;


    public BaseDatasetIterator(int batch, int numExamples, DataSetFetcher fetcher) {
        this.batch = batch;
        if (numExamples < 0)
            numExamples = fetcher.totalExamples();

        this.numExamples = numExamples;
        this.fetcher = fetcher;
    }

    @Override
    public boolean hasNext() {
        return fetcher.hasMore() && fetcher.cursor() < numExamples;
    }

    @Override
    public DataSet next() {
        fetcher.fetch(batch);
        return fetcher.next();
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        return fetcher.inputColumns();
    }

    @Override
    public int totalOutcomes() {
        return fetcher.totalOutcomes();
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
        fetcher.reset();
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
    public List<String> getLabels() {
        return null;
    }


    @Override
    public DataSet next(int num) {
        fetcher.fetch(num);
        DataSet next = fetcher.next();
        if (preProcessor != null)
            preProcessor.preProcess(next);
        return next;
    }


    @Override
    public DataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }
}
