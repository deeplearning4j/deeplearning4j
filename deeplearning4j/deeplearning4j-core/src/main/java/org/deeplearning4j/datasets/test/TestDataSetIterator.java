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

package org.deeplearning4j.datasets.test;

import lombok.Getter;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;

/**
 * Track number of times the dataset iterator has been called
 * @author agibsonccc
 *
 */
public class TestDataSetIterator implements DataSetIterator {
    /**
     * 
     */
    private static final long serialVersionUID = -3042802726018263331L;
    private DataSetIterator wrapped;
    private int numDataSets = 0;
    @Getter
    private DataSetPreProcessor preProcessor;


    public TestDataSetIterator(DataSetIterator wrapped) {
        super();
        this.wrapped = wrapped;
    }

    @Override
    public boolean hasNext() {
        return wrapped.hasNext();
    }

    @Override
    public DataSet next() {
        numDataSets++;
        DataSet next = wrapped.next();
        if (preProcessor != null)
            preProcessor.preProcess(next);
        return next;
    }

    @Override
    public void remove() {
        wrapped.remove();
    }

    @Override
    public int inputColumns() {
        return wrapped.inputColumns();
    }

    @Override
    public int totalOutcomes() {
        return wrapped.totalOutcomes();
    }

    @Override
    public boolean resetSupported() {
        return wrapped.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return wrapped.asyncSupported();
    }

    @Override
    public void reset() {
        wrapped.reset();
    }

    @Override
    public int batch() {
        return wrapped.batch();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }


    public synchronized int getNumDataSets() {
        return numDataSets;
    }

    @Override
    public DataSet next(int num) {
        return wrapped.next(num);
    }



}
