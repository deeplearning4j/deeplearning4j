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

package org.nd4j.parameterserver.updater;

import org.nd4j.aeron.ipc.NDArrayHolder;
import org.nd4j.parameterserver.updater.storage.InMemoryUpdateStorage;
import org.nd4j.parameterserver.updater.storage.UpdateStorage;

/**
 * Base class for the parameter updater
 * handling things such as update storage
 * and basic operations like reset and number of updates
 *
 * @author Adam Gibson
 */
public abstract class BaseParameterUpdater implements ParameterServerUpdater {
    protected UpdateStorage updateStorage;
    protected NDArrayHolder ndArrayHolder;

    public BaseParameterUpdater(UpdateStorage updateStorage, NDArrayHolder ndArrayHolder) {
        this.updateStorage = updateStorage;
        this.ndArrayHolder = ndArrayHolder;
    }

    /**
     * Returns true if the updater is
     * ready for a new array
     *
     * @return
     */
    @Override
    public boolean isReady() {
        return numUpdates() == requiredUpdatesForPass();
    }

    /**
     * Returns true if the
     * given updater is async
     * or synchronous
     * updates
     *
     * @return true if the given updater
     * is async or synchronous updates
     */
    @Override
    public boolean isAsync() {
        return true;
    }

    /**
     * Get the ndarray holder for this
     * updater
     *
     * @return the ndarray holder for this updater
     */
    @Override
    public NDArrayHolder ndArrayHolder() {
        return ndArrayHolder;
    }

    /**
     * Initialize this updater
     * with a custom update storage
     * @param updateStorage the update storage to use
     */
    public BaseParameterUpdater(UpdateStorage updateStorage) {
        this.updateStorage = updateStorage;
    }

    /**
     * Initializes this updater
     * with {@link InMemoryUpdateStorage}
     */
    public BaseParameterUpdater() {
        this(new InMemoryUpdateStorage());
    }



    /**
     * Reset internal counters
     * such as number of updates accumulated.
     */
    @Override
    public void reset() {
        updateStorage.clear();
    }


    /**
     * Num updates passed through
     * the updater
     *
     * @return the number of updates
     */
    @Override
    public int numUpdates() {
        return updateStorage.numUpdates();
    }
}
