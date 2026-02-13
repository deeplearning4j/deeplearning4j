/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.device;

import org.bytedeco.javacpp.Pointer;

/**
 * Default CPU no-op implementation of DeviceContextProvider.
 * Returns DeviceContext(0, null, null) for all methods.
 * switchDevice() is a no-op since CPU has no device switching.
 */
public class CpuDeviceContextProvider implements DeviceContextProvider {

    private static final DeviceContext CPU_CONTEXT = new DeviceContext(0, null, null);

    @Override
    public DeviceContext getCurrentContext() {
        return CPU_CONTEXT;
    }

    @Override
    public DeviceContext switchDevice(int deviceId, String caller, String reason) {
        return CPU_CONTEXT;
    }

    @Override
    public Pointer getFreshExecutionStream() {
        return null;
    }

    @Override
    public int getCurrentDeviceId() {
        return 0;
    }

    @Override
    public int getDeviceCount() {
        return 1;
    }

    @Override
    public void syncExecutionStream() {
        // no-op on CPU
    }

    @Override
    public boolean supportsStreams() {
        return false;
    }
}
