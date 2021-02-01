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

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.callbacks.DataSetCallback;

import java.util.concurrent.BlockingQueue;

/**
 * @deprecated Use {@link org.nd4j.linalg.dataset.AsyncDataSetIterator}
 */
@Slf4j
@Deprecated
public class AsyncDataSetIterator extends org.nd4j.linalg.dataset.AsyncDataSetIterator {

    /**
     * Create an Async iterator with the default queue size of 8
     * @param baseIterator Underlying iterator to wrap and fetch asynchronously from
     */
    public AsyncDataSetIterator(DataSetIterator baseIterator) {
        super(baseIterator);
    }

    /**
     * Create an Async iterator with the default queue size of 8
     * @param iterator Underlying iterator to wrap and fetch asynchronously from
     * @param queue    Queue size - number of iterators to
     */
    public AsyncDataSetIterator(DataSetIterator iterator, int queueSize, BlockingQueue<DataSet> queue) {
        super(iterator, queueSize, queue);
    }

    public AsyncDataSetIterator(DataSetIterator baseIterator, int queueSize) {
        super(baseIterator, queueSize);
    }

    public AsyncDataSetIterator(DataSetIterator baseIterator, int queueSize, boolean useWorkspace) {
        super(baseIterator, queueSize, useWorkspace);
    }

    public AsyncDataSetIterator(DataSetIterator baseIterator, int queueSize, boolean useWorkspace, Integer deviceId) {
        super(baseIterator, queueSize, useWorkspace, deviceId);
    }

    public AsyncDataSetIterator(DataSetIterator baseIterator, int queueSize, boolean useWorkspace, DataSetCallback callback) {
        super(baseIterator, queueSize, useWorkspace, callback);
    }

    public AsyncDataSetIterator(DataSetIterator iterator, int queueSize, BlockingQueue<DataSet> queue, boolean useWorkspace) {
        super(iterator, queueSize, queue, useWorkspace);
    }

    public AsyncDataSetIterator(DataSetIterator iterator, int queueSize, BlockingQueue<DataSet> queue, boolean useWorkspace, DataSetCallback callback) {
        super(iterator, queueSize, queue, useWorkspace, callback);
    }

    public AsyncDataSetIterator(DataSetIterator iterator, int queueSize, BlockingQueue<DataSet> queue,
                    boolean useWorkspace, DataSetCallback callback, Integer deviceId) {
        super(iterator, queueSize, queue, useWorkspace, callback, deviceId);
    }
}
