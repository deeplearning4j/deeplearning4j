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
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Single-process collective communicator that operates on local arrays.
 * <p>
 * This implementation works on any backend (CPU or GPU) and is the default
 * when NCCL is not available or when tensor parallelism size is 1.
 * <p>
 * For a single rank (tpSize=1), all operations are identity/no-ops:
 * <ul>
 *   <li>AllReduce: copy sendBuf to recvBuf (or no-op for in-place)</li>
 *   <li>AllGather: return a copy of the input</li>
 *   <li>Send/Recv: throw (no peer in single-rank group)</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class LocalCollectiveCommunicator implements CollectiveCommunicator {

    private final int numRanks;
    private final int rank;

    public LocalCollectiveCommunicator(int numRanks, int rank) {
        if (numRanks < 1) throw new IllegalArgumentException("numRanks must be >= 1");
        if (rank < 0 || rank >= numRanks) throw new IllegalArgumentException("rank out of range");
        this.numRanks = numRanks;
        this.rank = rank;
    }

    /**
     * Create a single-rank communicator (most common case for CPU or single-GPU).
     */
    public LocalCollectiveCommunicator() {
        this(1, 0);
    }

    @Override
    public int getNumRanks() {
        return numRanks;
    }

    @Override
    public int getRank() {
        return rank;
    }

    @Override
    public void allReduceSum(INDArray tensor) {
        // Single rank: no-op (tensor is already the full result)
    }

    @Override
    public void allReduce(INDArray sendBuf, INDArray recvBuf, int reduceOp) {
        if (sendBuf != recvBuf) {
            recvBuf.assign(sendBuf);
        }
    }

    @Override
    public INDArray allGather(INDArray localData) {
        return localData.dup();
    }

    @Override
    public INDArray allGatherAlongDim(INDArray localData, int dim) {
        return localData.dup();
    }

    @Override
    public void reduceScatter(INDArray sendBuf, INDArray recvBuf, int reduceOp) {
        if (sendBuf != recvBuf) {
            recvBuf.assign(sendBuf);
        }
    }

    @Override
    public void send(INDArray data, int peerRank) {
        throw new UnsupportedOperationException(
                "Point-to-point send is not supported in LocalCollectiveCommunicator. " +
                "Use NCCL or another distributed backend for multi-rank communication.");
    }

    @Override
    public void recv(INDArray recvBuf, int peerRank) {
        throw new UnsupportedOperationException(
                "Point-to-point recv is not supported in LocalCollectiveCommunicator. " +
                "Use NCCL or another distributed backend for multi-rank communication.");
    }

    @Override
    public String backendName() {
        return "Local";
    }

    @Override
    public void close() {
        // Nothing to clean up
    }

    @Override
    public String toString() {
        return "LocalCollectiveCommunicator{rank=" + rank + "/" + numRanks + "}";
    }
}
