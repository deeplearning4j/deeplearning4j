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

package org.nd4j.linalg.jcublas.buffer;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;


/**
 * Address retriever
 * for a data buffer (both on host and device)
 */
public class AddressRetriever {
    private static final AtomicAllocator allocator = AtomicAllocator.getInstance();

    /**
     * Retrieves the device pointer
     * for the given data buffer
     * @param buffer the buffer to get the device
     *               address for
     * @return the device address for the given
     * data buffer
     */
    public static long retrieveDeviceAddress(DataBuffer buffer, CudaContext context) {
        return allocator.getPointer(buffer, context).address();
    }


    /**
     * Returns the host address
     * @param buffer
     * @return
     */
    public static long retrieveHostAddress(DataBuffer buffer) {
        return allocator.getHostPointer(buffer).address();
    }

    /**
     * Retrieves the device pointer
     * for the given data buffer
     * @param buffer the buffer to get the device
     *               address for
     * @return the device pointer for the given
     * data buffer
     */
    public static Pointer retrieveDevicePointer(DataBuffer buffer, CudaContext context) {
        return allocator.getPointer(buffer, context);
    }


    /**
     * Returns the host pointer
     * @param buffer
     * @return
     */
    public static Pointer retrieveHostPointer(DataBuffer buffer) {
        return allocator.getHostPointer(buffer);
    }
}
