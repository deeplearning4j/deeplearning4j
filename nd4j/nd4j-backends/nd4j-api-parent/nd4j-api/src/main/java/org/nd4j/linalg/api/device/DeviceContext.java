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
 * Short-lived snapshot of a device context, containing the device ID and
 * fresh stream pointers. This must NOT be cached across device switches —
 * each switchDevice() or getCurrentContext() call returns a new instance
 * with valid pointers from the thread-local C++ ContextBuffers.
 *
 * On CPU backends, stream pointers are null.
 */
public final class DeviceContext {
    private final int deviceId;
    private final Pointer executionStream;
    private final Pointer copyStream;

    public DeviceContext(int deviceId, Pointer executionStream, Pointer copyStream) {
        this.deviceId = deviceId;
        this.executionStream = executionStream;
        this.copyStream = copyStream;
    }

    public int getDeviceId() {
        return deviceId;
    }

    public Pointer getExecutionStream() {
        return executionStream;
    }

    public Pointer getCopyStream() {
        return copyStream;
    }

    public boolean hasStreams() {
        return executionStream != null;
    }

    @Override
    public String toString() {
        return "DeviceContext{deviceId=" + deviceId +
                ", hasStreams=" + hasStreams() +
                (executionStream != null ? ", execStream=0x" + Long.toHexString(executionStream.address()) : "") +
                "}";
    }
}
