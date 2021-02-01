/*
 *  ******************************************************************************
 *  *
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
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.callbacks.DataSetCallback;

import java.util.concurrent.BlockingQueue;

/**
 * @deprecated Use {@link org.nd4j.linalg.dataset.AsyncMultiDataSetIterator}
 */
@Slf4j
@Deprecated
public class AsyncMultiDataSetIterator extends org.nd4j.linalg.dataset.AsyncMultiDataSetIterator {

    public AsyncMultiDataSetIterator(MultiDataSetIterator baseIterator) {
        super(baseIterator);
    }

    public AsyncMultiDataSetIterator(MultiDataSetIterator iterator, int queueSize, BlockingQueue<MultiDataSet> queue) {
        super(iterator, queueSize, queue);
    }

    public AsyncMultiDataSetIterator(MultiDataSetIterator baseIterator, int queueSize) {
        super(baseIterator, queueSize);
    }

    public AsyncMultiDataSetIterator(MultiDataSetIterator baseIterator, int queueSize, boolean useWorkspace) {
        super(baseIterator, queueSize, useWorkspace);
    }

    public AsyncMultiDataSetIterator(MultiDataSetIterator baseIterator, int queueSize, boolean useWorkspace,
                    Integer deviceId) {
        super(baseIterator, queueSize, useWorkspace, deviceId);
    }

    public AsyncMultiDataSetIterator(MultiDataSetIterator iterator, int queueSize, BlockingQueue<MultiDataSet> queue,
                    boolean useWorkspace) {
        super(iterator, queueSize, queue, useWorkspace);
    }

    public AsyncMultiDataSetIterator(MultiDataSetIterator iterator, int queueSize, BlockingQueue<MultiDataSet> queue,
                    boolean useWorkspace, DataSetCallback callback) {
        super(iterator, queueSize, queue, useWorkspace, callback);
    }

    public AsyncMultiDataSetIterator(MultiDataSetIterator iterator, int queueSize, BlockingQueue<MultiDataSet> queue,
                    boolean useWorkspace, DataSetCallback callback, Integer deviceId) {
        super(iterator, queueSize, queue, useWorkspace, callback, deviceId);
    }
}
