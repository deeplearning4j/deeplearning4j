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

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Closeable;

/**
 * Backend-agnostic interface for collective communication operations
 * used in tensor and pipeline parallelism.
 * <p>
 * Implementations include:
 * <ul>
 *   <li>{@link LocalCollectiveCommunicator} — single-process fallback (works on CPU and GPU)</li>
 *   <li>{@link NcclCommunicator} — NCCL-backed multi-GPU communication</li>
 * </ul>
 * <p>
 * Use {@link CollectiveCommunicatorFactory#create(TensorParallelConfig)} to obtain
 * the appropriate implementation for the current backend.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public interface CollectiveCommunicator extends Closeable {

    /** Reduce operation: sum */
    int REDUCE_SUM = 0;
    /** Reduce operation: product */
    int REDUCE_PROD = 1;
    /** Reduce operation: max */
    int REDUCE_MAX = 2;
    /** Reduce operation: min */
    int REDUCE_MIN = 3;
    /** Reduce operation: average */
    int REDUCE_AVG = 4;

    /**
     * @return the number of ranks in this communicator group
     */
    int getNumRanks();

    /**
     * @return this process's rank (0-indexed)
     */
    int getRank();

    /**
     * AllReduce sum in-place: sum tensor across all ranks, result stored in tensor.
     *
     * @param tensor the tensor to reduce (modified in-place)
     */
    void allReduceSum(INDArray tensor);

    /**
     * AllReduce with specified operation.
     *
     * @param sendBuf  send buffer
     * @param recvBuf  receive buffer (can be same as sendBuf for in-place)
     * @param reduceOp reduction operation constant (REDUCE_SUM, etc.)
     */
    void allReduce(INDArray sendBuf, INDArray recvBuf, int reduceOp);

    /**
     * AllGather: gather data from all ranks along the first dimension.
     * Each rank contributes its local data; the result has first dim multiplied by numRanks.
     *
     * @param localData this rank's data shard
     * @return gathered array with first dimension = localData.size(0) * numRanks
     */
    INDArray allGather(INDArray localData);

    /**
     * AllGather along a specific dimension.
     *
     * @param localData this rank's data shard
     * @param dim       dimension to gather along (supports negative indexing)
     * @return gathered array with specified dimension multiplied by numRanks
     */
    INDArray allGatherAlongDim(INDArray localData, int dim);

    /**
     * ReduceScatter: reduce then scatter across ranks.
     *
     * @param sendBuf  full data to reduce and scatter
     * @param recvBuf  this rank's shard of the reduced result
     * @param reduceOp reduction operation
     */
    void reduceScatter(INDArray sendBuf, INDArray recvBuf, int reduceOp);

    /**
     * Point-to-point send to a peer rank.
     *
     * @param data     data to send
     * @param peerRank destination rank
     */
    void send(INDArray data, int peerRank);

    /**
     * Point-to-point receive from a peer rank.
     *
     * @param recvBuf  buffer to receive into
     * @param peerRank source rank
     */
    void recv(INDArray recvBuf, int peerRank);

    /**
     * @return a human-readable name for this communicator type (e.g., "NCCL", "Local")
     */
    String backendName();

    /**
     * Close and release all resources held by this communicator.
     */
    @Override
    void close();
}
