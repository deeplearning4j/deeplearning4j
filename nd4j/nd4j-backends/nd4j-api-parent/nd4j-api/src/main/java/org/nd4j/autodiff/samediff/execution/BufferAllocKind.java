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

/**
 * Classification of how a buffer slot in an {@link ExecutionPlan} should be allocated.
 * Mirrors the allocation kinds used by ONNX Runtime's AllocationPlanner:
 * <ul>
 *   <li>{@link #PRE_EXISTING} — Placeholder inputs owned by the caller</li>
 *   <li>{@link #CONSTANT} — Graph constants, already allocated in the SameDiff instance</li>
 *   <li>{@link #VARIABLE} — Trainable variables, already allocated</li>
 *   <li>{@link #ALLOCATE} — Fresh intermediate allocation from the workspace</li>
 *   <li>{@link #REUSE} — Reuses a dead buffer whose last consumer has finished</li>
 *   <li>{@link #OUTPUT} — Graph output that must survive beyond workspace scope</li>
 * </ul>
 */
public enum BufferAllocKind {
    /** Placeholder/feed — caller owns the memory, not managed by the plan. */
    PRE_EXISTING,

    /** Graph constant — already allocated in the SameDiff instance. */
    CONSTANT,

    /** Trainable variable — already allocated in the SameDiff instance. */
    VARIABLE,

    /** Fresh intermediate allocation from the workspace pool. */
    ALLOCATE,

    /** Reuses a previously freed buffer at the same offset (liveness-based). */
    REUSE,

    /** Graph output — allocated from workspace but must be detached before return. */
    OUTPUT
}
