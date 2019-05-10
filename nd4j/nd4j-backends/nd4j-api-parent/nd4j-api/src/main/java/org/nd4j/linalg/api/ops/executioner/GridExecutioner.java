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

package org.nd4j.linalg.api.ops.executioner;

import org.nd4j.linalg.api.ops.aggregates.Aggregate;

/**
 * @author raver119@gmail.com
 */
public interface GridExecutioner extends OpExecutioner {

    /**
     * This method forces all currently enqueued ops to be executed immediately
     *
     * PLEASE NOTE: This call CAN be non-blocking, if specific backend implementation supports that.
     */
    void flushQueue();

    /**
     * This method forces all currently enqueued ops to be executed immediately
     *
     * PLEASE NOTE: This call is always blocking, until all queued operations are finished
     */
    void flushQueueBlocking();


    /**
     * This method returns number of operations currently enqueued for execution
     *
     * @return
     */
    int getQueueLength();


    /**
     * This method enqueues aggregate op for future invocation
     *
     * @param op
     */
    void aggregate(Aggregate op);

    /**
     * This method enqueues aggregate op for future invocation.
     * Key value will be used to batch individual ops
     *
     * @param op
     * @param key
     */
    void aggregate(Aggregate op, long key);
}
