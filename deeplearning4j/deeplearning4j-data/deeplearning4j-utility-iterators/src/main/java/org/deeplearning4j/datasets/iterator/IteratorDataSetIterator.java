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

package org.deeplearning4j.datasets.iterator;


import lombok.Getter;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.*;

/**
 * A DataSetIterator that works on an Iterator<DataSet>, combining and splitting the input DataSet objects as
 * required to get a consistent batch size.
 *
 * Typically used in Spark training, but may be used elsewhere.
 * NOTE: reset method is not supported here.
 */
public class IteratorDataSetIterator implements DataSetIterator {

    private final Iterator<DataSet> iterator;
    private final int batchSize;
    private final LinkedList<DataSet> queued; //Used when splitting larger examples than we want to return in a batch
    @Getter
    private DataSetPreProcessor preProcessor;

    private int inputColumns = -1;
    private int totalOutcomes = -1;

    private int cursor = 0;

    public IteratorDataSetIterator(Iterator<DataSet> iterator, int batchSize) {
        this.iterator = iterator;
        this.batchSize = batchSize;
        this.queued = new LinkedList<>();
    }

    @Override
    public boolean hasNext() {
        return !queued.isEmpty() || iterator.hasNext();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public DataSet next(int num) {
        if (!hasNext())
            throw new NoSuchElementException();

        List<DataSet> list = new ArrayList<>();
        int countSoFar = 0;
        while ((!queued.isEmpty() || iterator.hasNext()) && countSoFar < batchSize) {
            DataSet next;
            if (!queued.isEmpty()) {
                next = queued.removeFirst();
            } else {
                next = iterator.next();
            }
            int nExamples = next.numExamples();
            if (countSoFar + nExamples <= batchSize) {
                //Add the entire DataSet as-is
                list.add(next);
            } else {
                //Otherwise, split it
                DataSet toKeep = (DataSet) next.getRange(0, batchSize - countSoFar);
                DataSet toCache = (DataSet) next.getRange(batchSize - countSoFar, nExamples);
                list.add(toKeep);
                queued.add(toCache);
            }

            countSoFar += nExamples;
        }

        if (inputColumns == -1) {
            //Set columns etc for later use
            DataSet temp = list.get(0);

            // FIXME: int cast
            inputColumns = (int) temp.getFeatures().size(1);
            totalOutcomes = temp.getLabels() == null ? 0 : (int) temp.getLabels().size(1); //May be null for layerwise pretraining
        }

        DataSet out;
        if (list.size() == 1) {
            out = list.get(0);
        } else {
            out = DataSet.merge(list);
        }

        if (preProcessor != null) {
            if (!out.isPreProcessed()) {
                preProcessor.preProcess(out);
                out.markAsPreProcessed();
            }
        }
        cursor += out.numExamples();
        return out;
    }

    @Override
    public int inputColumns() {
        if (inputColumns != -1)
            return inputColumns;
        prefetchBatchSetInputOutputValues();
        return inputColumns;
    }

    @Override
    public int totalOutcomes() {
        if (totalOutcomes != -1)
            return totalOutcomes;
        prefetchBatchSetInputOutputValues();
        return totalOutcomes;
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
        throw new UnsupportedOperationException("Reset not supported");
    }

    @Override
    public int batch() {
        return batchSize;
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
    public void remove() {
        throw new UnsupportedOperationException("Not supported");
    }

    private void prefetchBatchSetInputOutputValues() {
        if (!iterator.hasNext())
            return;
        DataSet next = iterator.next();
        inputColumns = (int) next.getFeatures().size(1);
        totalOutcomes = (int) next.getLabels().size(1);
        queued.add(next);
    }
}
