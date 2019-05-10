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
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Base class for common logic in update storage
 *
 * @author Adam Gibson
 */
public abstract class BaseUpdateStorage implements UpdateStorage {
    /**
     * Get the update at the specified index
     *
     * @param index the update to get
     * @return the update at the specified index
     */
    @Override
    public NDArrayMessage getUpdate(int index) {
        if (index >= numUpdates())
            throw new IndexOutOfBoundsException(
                            "Index passed in " + index + " was >= current number of updates " + numUpdates());
        return doGetUpdate(index);
    }

    /**
     * A method for actually performing the implementation
     * of retrieving the ndarray
     * @param index the index of the {@link INDArray} to get
     * @return the ndarray at the specified index
     */
    public abstract NDArrayMessage doGetUpdate(int index);

    /**
     * Close the database
     */
    @Override
    public void close() {
        //default no op
    }
}
