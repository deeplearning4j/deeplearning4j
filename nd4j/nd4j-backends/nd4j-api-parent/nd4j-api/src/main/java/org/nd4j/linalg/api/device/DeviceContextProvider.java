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
 * SPI interface for device context management. Backends (CUDA, CPU) provide
 * implementations that handle device switching and stream lifecycle.
 *
 * The CUDA implementation consolidates the logic that was previously scattered
 * across CudaAffinityManager.unsafeSetDevice(), CudaZeroHandler.getCudaContext(),
 * and DynamicShapePlanExecutor.getFreshExecStream() into a single place.
 */
public interface DeviceContextProvider {

    /**
     * Get context for current thread's device with fresh stream pointers.
     * Always goes through defaultLaunchContext() to get valid pointers from
     * the thread-local C++ ContextBuffers.
     */
    DeviceContext getCurrentContext();

    /**
     * Switch to target device and return fresh context with valid streams.
     * This is the SINGLE entry point for all device switching in the system.
     *
     * @param deviceId target device ID
     * @param caller   class/method performing the switch (for tracing)
     * @param reason   why the switch is happening (for tracing)
     * @return fresh DeviceContext with valid stream pointers
     */
    DeviceContext switchDevice(int deviceId, String caller, String reason);

    /**
     * Get a fresh execution stream pointer for the current device.
     * Equivalent to getCurrentContext().getExecutionStream() but avoids
     * allocating a DeviceContext object when only the stream is needed.
     */
    Pointer getFreshExecutionStream();

    /**
     * Get current thread's device ID without switching.
     */
    int getCurrentDeviceId();

    /**
     * Get total number of available devices.
     */
    int getDeviceCount();

    /**
     * Synchronize the current execution stream.
     */
    void syncExecutionStream();

    /**
     * Whether this backend supports GPU streams.
     * Returns false for CPU backends.
     */
    boolean supportsStreams();
}
