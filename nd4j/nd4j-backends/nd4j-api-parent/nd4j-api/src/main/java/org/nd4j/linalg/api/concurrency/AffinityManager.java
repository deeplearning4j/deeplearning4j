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

package org.nd4j.linalg.api.concurrency;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
public interface AffinityManager {

    enum Location {
        HOST, DEVICE, EVERYWHERE,
    }

    /**
     * This method returns deviceId for current thread
     * @return
     */
    Integer getDeviceForCurrentThread();

    /**
     * This method returns deviceId for specified thread
     * @param thread
     * @return
     */
    Integer getDeviceForThread(Thread thread);

    /**
     * This method returns deviceId for specified threadId
     *
     * @param threadId
     * @return
     */
    Integer getDeviceForThread(long threadId);

    /**
     * This method returns id of current device for a given INDArray
     *
     * @param array
     * @return
     */
    Integer getDeviceForArray(INDArray array);

    /**
     * This method attaches specified thread to specified device
     *
     * @param thread
     * @param deviceId
     */
    void attachThreadToDevice(Thread thread, Integer deviceId);


    /**
     * This method attaches specified thread (by Id) to specified device
     *
     * @param threadId java ID of the thread
     * @param deviceId
     */
    void attachThreadToDevice(long threadId, Integer deviceId);

    /**
     * This method returns number of available devices
     * @return
     */
    int getNumberOfDevices();

    /**
     * Utility method, to associate INDArray with specific device (backend-specific)
     *
     * @param array
     */
    void touch(INDArray array);

    /**
     * Utility method, to associate INDArray with specific device (backend-specific)
     * 
     * @param buffer
     */
    void touch(DataBuffer buffer);

    /**
     * This method replicates given INDArray, and places it to target device.
     *
     * @param deviceId  target deviceId
     * @param array INDArray to replicate
     * @return
     */
    INDArray replicateToDevice(Integer deviceId, INDArray array);

    /**
     * This method replicates given DataBuffer, and places it to target device.
     *
     * @param deviceId  target deviceId
     * @param buffer
     * @return
     */
    DataBuffer replicateToDevice(Integer deviceId, DataBuffer buffer);

    /**
     * This method tags specific INDArray as "recent" on specified location
     *
     * @param location
     */
    void tagLocation(INDArray array, Location location);

    /**
     * This method tags specific DataBuffer as "recent" on specified location
     *
     * @param location
     */
    void tagLocation(DataBuffer buffer, Location location);


    /**
     * This method propagates given INDArray to specified location
     *
     * @param array
     * @param location
     */
    void ensureLocation(INDArray array, Location location);

    /**
     * This method returns last-updated location for the given INDArray
     * @param array
     * @return
     */
    Location getActiveLocation(INDArray array);

    /**
     * This method forces specific device for current thread.
     *
     * PLEASE NOTE: This method is UNSAFE and should NOT be used with 100% clearance about it.
     *
     * @param deviceId
     */
    void unsafeSetDevice(Integer deviceId);


    /**
     * This method returns TRUE if cross-device access is allowed on this system
     */
    boolean isCrossDeviceAccessSupported();

    /**
     * This method allows to block cross-device access. Mostly suitable for debugging/testing purposes
     *
     * @param reallyAllow
     */
    void allowCrossDeviceAccess(boolean reallyAllow);
}
