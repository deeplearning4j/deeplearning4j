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

import org.nd4j.aeron.ipc.NDArrayMessage;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Created by agibsonccc on 12/1/16.
 */
public class SoftSyncParameterUpdater extends BaseParameterUpdater {
    //track time stamps of messages coming in to find out which generation a message is meant for
    //alxways log where the message time stamp began
    private Map<Long, Integer> timeStampsForGeneration;
    //s is the number of updates
    private int s;
    private int currentVersion;
    private int accumulatedUpdates = 0;
    private double scalingFactor;


    /**
     * Returns the number of required
     * updates for a new pass
     *
     * @return the number of required updates for a new pass
     */
    @Override
    public int requiredUpdatesForPass() {
        return 0;
    }

    /**
     * Returns the current status of this parameter server
     * updater
     *
     * @return
     */
    @Override
    public Map<String, Number> status() {
        return null;
    }

    /**
     * Serialize this updater as json
     *
     * @return
     */
    @Override
    public String toJson() {
        return null;
    }

    /**
     * Reset internal counters
     * such as number of updates accumulated.
     */
    @Override
    public void reset() {
        currentVersion++;
    }

    /**
     * Returns true if
     * the updater has accumulated enough ndarrays to
     * replicate to the workers
     *
     * @return true if replication should happen,false otherwise
     */
    @Override
    public boolean shouldReplicate() {
        return accumulatedUpdates == s;
    }

    /**
     * Do an update based on the ndarray message.
     *
     * @param message
     */
    @Override
    public void update(NDArrayMessage message) {

    }

    /**
     * Updates result
     * based on arr along a particular
     * {@link INDArray#tensorAlongDimension(int, int...)}
     *
     * @param arr        the array to update
     * @param result     the result ndarray to update
     * @param idx        the index to update
     * @param dimensions the dimensions to update
     */
    @Override
    public void partialUpdate(INDArray arr, INDArray result, long idx, int... dimensions) {

    }

    /**
     * Updates result
     * based on arr
     *
     * @param arr    the array to update
     * @param result the result ndarray to update
     */
    @Override
    public void update(INDArray arr, INDArray result) {

    }
}
