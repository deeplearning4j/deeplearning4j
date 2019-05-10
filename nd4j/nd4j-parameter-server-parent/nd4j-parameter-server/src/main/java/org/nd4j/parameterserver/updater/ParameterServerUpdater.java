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
import org.nd4j.aeron.ipc.NDArrayMessage;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * A parameter server updater
 * for applying updates on the parameter server
 *
 * @author Adam Gibson
 */
public interface ParameterServerUpdater {

    /**
     * Returns the number of required
     * updates for a new pass
     * @return the number of required updates for a new pass
     */
    int requiredUpdatesForPass();

    /**
     * Returns true if the updater is
     * ready for a new array
     * @return
     */
    boolean isReady();

    /**
     * Returns true if the
     * given updater is async
     * or synchronous
     * updates
     * @return true if the given updater
     * is async or synchronous updates
     */
    boolean isAsync();

    /**
     * Get the ndarray holder for this
     * updater
     * @return the ndarray holder for this updater
     */
    NDArrayHolder ndArrayHolder();

    /**
     * Num updates passed through
     * the updater
     * @return the number of updates
     *
     */
    int numUpdates();


    /**
     * Returns the current status of this parameter server
     * updater
     * @return
     */
    Map<String, Number> status();

    /**
     * Serialize this updater as json
     * @return
     */
    String toJson();

    /**
     * Reset internal counters
     * such as number of updates accumulated.
     */
    void reset();

    /**
     * Returns true if
     * the updater has accumulated enough ndarrays to
     * replicate to the workers
     * @return true if replication should happen,false otherwise
     */
    boolean shouldReplicate();

    /**
     * Do an update based on the ndarray message.
     * @param message
     */
    void update(NDArrayMessage message);

    /**
     * Updates result
     * based on arr along a particular
     * {@link INDArray#tensorAlongDimension(int, int...)}
     * @param arr the array to update
     * @param result the result ndarray to update
     * @param idx the index to update
     * @param dimensions the dimensions to update
     */
    void partialUpdate(INDArray arr, INDArray result, long idx, int... dimensions);

    /**
     * Updates result
     * based on arr
     * @param arr the array to update
     * @param result the result ndarray to update
     */
    void update(INDArray arr, INDArray result);
}
