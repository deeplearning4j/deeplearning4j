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

package org.deeplearning4j.datasets.iterator;

import lombok.Getter;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;

import java.util.List;

public class BaseDatasetIterator implements DataSetIterator {

    protected int batch, numExamples;
    protected BaseDataFetcher fetcher;
    @Getter
    protected DataSetPreProcessor preProcessor;


    public BaseDatasetIterator(int batch, int numExamples, BaseDataFetcher fetcher) {
        if(batch <= 0){
            throw new IllegalArgumentException("Invalid minibatch size: must be > 0 (got: " + batch + ")");
        }
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
        DataSet result = fetcher.next();
        if (preProcessor != null) {
            preProcessor.preProcess(result);
        }
        return result;
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
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }


}
