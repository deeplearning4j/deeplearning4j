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

import lombok.extern.slf4j.Slf4j;

/**
 * Factory for creating the appropriate {@link CollectiveCommunicator} based on
 * the current backend and configuration.
 * <p>
 * Selection logic:
 * <ol>
 *   <li>If TP size is 1, always returns {@link LocalCollectiveCommunicator}</li>
 *   <li>If NCCL is available (CUDA backend built with NCCL), returns {@link NcclCommunicator}</li>
 *   <li>Otherwise, returns {@link LocalCollectiveCommunicator} with a warning</li>
 * </ol>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class CollectiveCommunicatorFactory {

    private CollectiveCommunicatorFactory() {
    }

    /**
     * Create a collective communicator appropriate for the given config.
     *
     * @param config tensor parallel configuration
     * @return a CollectiveCommunicator instance
     */
    public static CollectiveCommunicator create(TensorParallelConfig config) {
        if (!config.isEnabled()) {
            return new LocalCollectiveCommunicator();
        }

        // Try NCCL first for multi-GPU
        if (NcclCommunicator.isAvailable()) {
            log.info("Creating NCCL communicator for TP rank {}/{} on device {}",
                    config.getRank(), config.getTensorParallelSize(), config.myDeviceId());
            return NcclCommunicator.create(
                    config.getTensorParallelSize(),
                    config.getRank(),
                    config.myDeviceId());
        }

        // Fallback to local communicator
        log.warn("NCCL is not available. Multi-GPU tensor parallelism will use local " +
                 "communicator (no cross-GPU communication). Build with -DHELPERS_nccl=ON " +
                 "for NCCL support.");
        return new LocalCollectiveCommunicator(config.getTensorParallelSize(), config.getRank());
    }

    /**
     * Create a local communicator (always available, any backend).
     *
     * @param numRanks total ranks
     * @param rank     this rank
     * @return a LocalCollectiveCommunicator
     */
    public static CollectiveCommunicator createLocal(int numRanks, int rank) {
        return new LocalCollectiveCommunicator(numRanks, rank);
    }

    /**
     * Check if a distributed (non-local) communicator is available.
     */
    public static boolean isDistributedAvailable() {
        return NcclCommunicator.isAvailable();
    }
}
