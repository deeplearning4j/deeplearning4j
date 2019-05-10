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
public abstract class BasicAffinityManager implements AffinityManager {
    @Override
    public Integer getDeviceForCurrentThread() {
        return 0;
    }

    @Override
    public Integer getDeviceForThread(Thread thread) {
        return 0;
    }

    @Override
    public Integer getDeviceForThread(long threadId) {
        return 0;
    }

    @Override
    public void attachThreadToDevice(Thread thread, Integer deviceId) {
        // no-op
    }

    @Override
    public void attachThreadToDevice(long threadId, Integer deviceId) {
        // no-op
    }

    @Override
    public Integer getDeviceForArray(INDArray array) {
        return 0;
    }

    @Override
    public int getNumberOfDevices() {
        return 1;
    }

    /**
     * This method replicates given INDArray, and places it to target device.
     *
     * @param deviceId target deviceId
     * @param array    INDArray to replicate
     * @return
     */
    @Override
    public INDArray replicateToDevice(Integer deviceId, INDArray array) {
        return null;
    }

    /**
     * This method replicates given DataBuffer, and places it to target device.
     *
     * @param deviceId target deviceId
     * @param buffer
     * @return
     */
    @Override
    public DataBuffer replicateToDevice(Integer deviceId, DataBuffer buffer) {
        return null;
    }

    @Override
    public void tagLocation(INDArray array, Location location) {
        // no-op
    }

    @Override
    public void tagLocation(DataBuffer buffer, Location location) {
        // no-op
    }

    @Override
    public void unsafeSetDevice(Integer deviceId) {
        // no-op
    }

    @Override
    public void ensureLocation(INDArray array, Location location) {
        // no-op
    }

    @Override
    public boolean isCrossDeviceAccessSupported() {
        return true;
    }

    @Override
    public void allowCrossDeviceAccess(boolean reallyAllow) {
        // no-op
    }

    @Override
    public Location getActiveLocation(INDArray array) {
        return Location.EVERYWHERE;
    }
}
