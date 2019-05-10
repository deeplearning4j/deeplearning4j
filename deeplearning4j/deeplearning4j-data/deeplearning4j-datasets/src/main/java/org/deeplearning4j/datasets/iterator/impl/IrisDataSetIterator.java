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

import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;

/**
 * IrisDataSetIterator: An iterator for the well-known Iris dataset. 4 features, 3 label classes<br>
 * <a href="https://archive.ics.uci.edu/ml/datasets/Iris">https://archive.ics.uci.edu/ml/datasets/Iris</a>
 */
public class IrisDataSetIterator extends BaseDatasetIterator {

    /**
     * 
     */
    private static final long serialVersionUID = -2022454995728680368L;

    /**
     * Create an iris iterator for full batch training - i.e., all 150 examples are included per minibatch
     */
    public IrisDataSetIterator(){
        this(150, 150);
    }

    /**
     * IrisDataSetIterator handles traversing through the Iris Data Set.
     * @see <a href="https://archive.ics.uci.edu/ml/datasets/Iris">https://archive.ics.uci.edu/ml/datasets/Iris</a>
     *
     * @param batch Batch size
     * @param numExamples Total number of examples
     */
    public IrisDataSetIterator(int batch, int numExamples) {
        super(batch, numExamples, new IrisDataFetcher());
    }


    @Override
    public DataSet next() {
        fetcher.fetch(batch);
        DataSet next = fetcher.next();
        if(preProcessor != null) {
            preProcessor.preProcess(next);
        }

        return next;
    }
}
