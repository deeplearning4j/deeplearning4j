/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.datasets.iterator;

import lombok.val;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

public class ScrollableDataSetIterator implements DataSetIterator {
    private int thisPart = 0;
    private int top = 0;
    private int bottom = 0;
    protected DataSetIterator backedIterator;
    protected AtomicLong counter = new AtomicLong(0);

    protected AtomicBoolean resetPending = new AtomicBoolean(false);
    protected DataSet firstTrain = null;
    protected MultiDataSet firstMultiTrain = null;
    private double ratio;
    private long totalExamples;
    private long itemsPerPart;
    private long current;


    public ScrollableDataSetIterator(int num, DataSetIterator backedIterator, AtomicLong counter,
                                     AtomicBoolean resetPending, DataSet firstTrain, double ratio,
                                     int totalExamples) {
        this.thisPart = num;
        this.backedIterator = backedIterator;
        this.counter = counter;
        this.resetPending = resetPending;
        this.firstTrain = firstTrain;
        this.ratio = ratio;
        this.totalExamples = totalExamples;
        this.itemsPerPart = (long)(totalExamples * ratio);
        this.current = 0;
    }

    public ScrollableDataSetIterator(int num, DataSetIterator backedIterator, AtomicLong counter,
                                     AtomicBoolean resetPending, DataSet firstTrain,
                                     int[] itemsPerPart) {
        this.thisPart = num;
        this.bottom = itemsPerPart[0];
        this.top = bottom + itemsPerPart[1];
        this.itemsPerPart = top;

        this.backedIterator = backedIterator;
        this.counter = counter;
        //this.resetPending = resetPending;
        this.firstTrain = firstTrain;
        //this.totalExamples = totalExamples;
        this.current = 0;
    }

    @Override
    public DataSet next(int i) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return backedIterator.getLabels();
    }

    @Override
    public int inputColumns() {
        return backedIterator.inputColumns();
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int totalOutcomes() {
        return backedIterator.totalOutcomes();
    }

    @Override
    public boolean resetSupported() {
        return backedIterator.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return backedIterator.asyncSupported();
    }

    @Override
    public void reset() {
        resetPending.set(true);
    }

    @Override
    public int batch() {
        return backedIterator.batch();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        backedIterator.setPreProcessor(dataSetPreProcessor);
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {

        return backedIterator.getPreProcessor();
    }


    @Override
    public boolean hasNext() {
        if (resetPending.get()) {
            if (resetSupported()) {
                backedIterator.reset();
                counter.set(0);
                current = 0;
                resetPending.set(false);
            } else
                throw new UnsupportedOperationException("Reset isn't supported by underlying iterator");
        }

        boolean state = false;
        if (current >= top)
            return false;
        state = backedIterator.hasNext();
        if (!state)
            return false;
        if (state && counter.get() < itemsPerPart)
            return true;
        else
            return false;

    }

    @Override
    public DataSet next() {
        counter.incrementAndGet();
        if ((current == 0) && (bottom != 0)) {
            backedIterator.reset();
            long cnt = current;
            for (; cnt < bottom; ++cnt) {
                if (backedIterator.hasNext())
                    backedIterator.next();
            }
            current = cnt+1;
        }
        else current++;
        val p = backedIterator.next();
        return p;
    }
}
