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

package org.nd4j.autodiff.samediff.execution;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.buffer.DataType;

/**
 * Describes how a single intermediate tensor is placed within the pre-allocated
 * workspace buffer of an {@link ExecutionPlan}.
 *
 * <p>For {@link BufferAllocKind#ALLOCATE} and {@link BufferAllocKind#OUTPUT} kinds,
 * {@code offsetInPool} is the byte offset from the start of the workspace where
 * this tensor's data begins.  For {@link BufferAllocKind#REUSE}, the offset is
 * inherited from the buffer being reused (identified by {@code reusesBufferIndex}).</p>
 */
@Data
@AllArgsConstructor
public class BufferAllocation {
    /** Unique index of this buffer slot in the plan. */
    private final int index;

    /** Byte offset into the contiguous workspace pool. */
    private final long offsetInPool;

    /** Size of this allocation in bytes. */
    private final long sizeInBytes;

    /** Data type of the tensor stored here. */
    private final DataType dataType;

    /** Shape of the tensor stored here. */
    private final long[] shape;

    /**
     * If {@code kind == REUSE}, the index of the original buffer being reused.
     * Otherwise {@code -1}.
     */
    private final int reusesBufferIndex;

    /** How this buffer should be managed. */
    private final BufferAllocKind kind;

    /** The variable name this buffer is associated with. */
    private final String variableName;
}
