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

package org.nd4j.parameterserver.updater.storage;

import org.nd4j.aeron.ipc.NDArrayMessage;

/**
 * An interface for storing parameter server updates.
 * This is used by an {@link org.nd4j.parameterserver.updater.ParameterServerUpdater}
 * to handle storage of ndarrays
 *
 * @author Adam Gibson
 */
public interface UpdateStorage {

    /**
     * Add an ndarray to the storage
     * @param array the array to add
     */
    void addUpdate(NDArrayMessage array);

    /**
     * The number of updates added
     * to the update storage
     * @return
     */
    int numUpdates();

    /**
     * Clear the array storage
     */
    void clear();

    /**
     * Get the update at the specified index
     * @param index the update to get
     * @return the update at the specified index
     */
    NDArrayMessage getUpdate(int index);

    /**
     * Close the database
     */
    void close();

}
