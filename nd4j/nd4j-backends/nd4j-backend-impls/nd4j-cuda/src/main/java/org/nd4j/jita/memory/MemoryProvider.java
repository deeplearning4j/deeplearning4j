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

package org.nd4j.jita.memory;

import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.pointers.PointersPair;

/**
 * This interface describes 2 basic methods to work with memory: malloc and free.
 *
 * @author raver119@gmail.com
 */
public interface MemoryProvider {
    /**
     * This method provides PointersPair to memory chunk specified by AllocationShape
     *
     * @param shape shape of desired memory chunk
     * @param point target AllocationPoint structure
     * @param location either HOST or DEVICE
     * @return
     */
    PointersPair malloc(AllocationShape shape, AllocationPoint point, AllocationStatus location);

    /**
     * This method frees specific chunk of memory, described by AllocationPoint passed in
     *
     * @param point
     */
    void free(AllocationPoint point);

    /**
     * This method checks specified device for specified amount of memory
     *
     * @param deviceId
     * @param requiredMemory
     * @return
     */
    boolean pingDeviceForFreeMemory(Integer deviceId, long requiredMemory);


    void purgeCache();
}
