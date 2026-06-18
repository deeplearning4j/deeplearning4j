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

package org.nd4j.linalg.framework.device;

/**
 * Reason for a device transfer.
 *
 * @author ND4J Team
 */
public enum TransferReason {
    /**
     * Weight/constant migration to execution device
     */
    CONSTANT_REPLICATION,

    /**
     * Input array moved to execution device
     */
    EXTERNAL_INPUT_MIGRATION,

    /**
     * KV cache cross-device scatter
     */
    KV_CACHE_SCATTER,

    /**
     * .dup() before cross-device copy
     */
    VIEW_DUP_FOR_REPLICATION,

    /**
     * Pin policy enforcement
     */
    DEVICE_AFFINITY_PIN,

    /**
     * OOM fallback to host/other device
     */
    FALLBACK_MIGRATION,

    /**
     * D2D into CUDA graph capture buffer
     */
    CAPTURE_BUFFER_COPY,

    /**
     * D2H for host-side read
     */
    SYNC_TO_HOST,

    /**
     * H2D for device-side compute
     */
    SYNC_TO_DEVICE
}
