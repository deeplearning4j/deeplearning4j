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

package org.nd4j.linalg.dataset;

import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;

/**
 * Iterate over a dataset
 * with views
 *
 * @author Adam Gibson
 */
public class ViewIterator implements DataSetIterator {
    private int batchSize = -1;
    private int cursor = 0;
    private DataSet data;
    private DataSetPreProcessor preProcessor;

    public ViewIterator(DataSet data, int batchSize) {
        this.batchSize = batchSize;
        this.data = data;
    }

    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException("Only allowed to retrieve dataset based on batch size");
    }

    @Override
    public int inputColumns() {
        return data.numInputs();
    }

    @Override
    public int totalOutcomes() {
        return data.numOutcomes();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        //Already all in memory
        return false;
    }

    @Override
    public void reset() {
        cursor = 0;
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
    public DataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return cursor < data.numExamples();
    }

    @Override
    public void remove() {}

    @Override
    public DataSet next() {
        int last = Math.min(data.numExamples(), cursor + batch());
        DataSet next = (DataSet) data.getRange(cursor, last);
        if (preProcessor != null)
            preProcessor.preProcess(next);
        cursor += batch();
        return next;
    }
}
