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

package org.nd4j.linalg.dataset.api.iterator;

import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * @author raver119@gmail.com
 */
public interface ParallelMultiDataSetIterator extends MultiDataSetIterator {

    /**
     * This method sets consumer affinity to specific producer
     *
     * PLEASE NOTE: this method is optional, and it'll change only nextFor()/hasNextFor() mechanics
     */
    void attachThread(int producer);

    /**
     * Returns true, if attached producer has something in queue, false otherwise
     *
     * @return
     */
    boolean hasNextFor();

    /**
     * Returns true, if attached producer has something in queue, false otherwise
     *
     * @param consumer
     * @return
     */
    boolean hasNextFor(int consumer);

    /**
     * Returns next DataSet for given consumer
     *
     * @param consumer
     * @return
     */
    MultiDataSet nextFor(int consumer);

    /**
     * Returns next DataSet for attached consumer
     *
     * @return
     */
    MultiDataSet nextFor();
}
