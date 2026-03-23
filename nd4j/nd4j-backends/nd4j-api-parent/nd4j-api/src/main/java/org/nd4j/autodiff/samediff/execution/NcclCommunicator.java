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

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/**
 * Java wrapper around NCCL collective communication operations.
 * <p>
 * Manages the lifecycle of an NCCL communicator and provides typed
 * collective operations (AllReduce, AllGather, ReduceScatter, Send, Recv)
 * on {@link INDArray} objects.
 * <p>
 * Usage:
 * <pre>
 *   // Single-process multi-GPU
 *   NcclCommunicator comm = NcclCommunicator.create(numGpus, rank, deviceId);
 *
 *   // AllReduce sum
 *   comm.allReduceSum(tensor);
 *
 *   // AllGather
 *   INDArray gathered = comm.allGather(localShard);
 *
 *   comm.close();
 * </pre>
 *
 * Adam Gibson
 * @see TensorParallelRunner
 */
@Slf4j
public class NcclCommunicator implements CollectiveCommunicator {

    /** Reduce operation constants matching C++ enum */
    public static final int REDUCE_SUM = 0;
    public static final int REDUCE_PROD = 1;
    public static final int REDUCE_MAX = 2;
    public static final int REDUCE_MIN = 3;
    public static final int REDUCE_AVG = 4;

    @Getter
    private final int numRanks;

    @Getter
    private final int rank;

    @Getter
    private final int deviceId;

    private Pointer commHandle;
    private boolean closed = false;

    private NcclCommunicator(int numRanks, int rank, int deviceId, Pointer commHandle) {
        this.numRanks = numRanks;
        this.rank = rank;
        this.deviceId = deviceId;
        this.commHandle = commHandle;
    }

    /**
     * Create an NCCL communicator for single-process multi-GPU.
     *
     * @param numRanks  number of GPUs in the group
     * @param rank      this GPU's rank (0-indexed)
     * @param deviceId  CUDA device ID
     * @return the communicator
     * @throws IllegalStateException if NCCL is not available
     */
    public static NcclCommunicator create(int numRanks, int rank, int deviceId) {
        Pointer handle = Nd4j.getNativeOps().ncclCommInit(numRanks, rank, deviceId);
        if (handle == null) {
            throw new IllegalStateException(
                    "Failed to initialize NCCL communicator. Ensure NCCL is installed and " +
                    "the native library was built with -DHELPERS_nccl=ON");
        }
        log.info("NCCL communicator initialized: rank {}/{} on device {}", rank, numRanks, deviceId);
        return new NcclCommunicator(numRanks, rank, deviceId, handle);
    }

    /**
     * Create NCCL communicators for all GPUs in a single process.
     *
     * @param deviceIds array of CUDA device IDs to use
     * @return array of communicators, one per device
     */
    public static NcclCommunicator[] createAll(int[] deviceIds) {
        int n = deviceIds.length;
        NcclCommunicator[] comms = new NcclCommunicator[n];
        for (int i = 0; i < n; i++) {
            comms[i] = create(n, i, deviceIds[i]);
        }
        return comms;
    }

    /**
     * AllReduce sum: sum tensor across all ranks, result on all ranks.
     * <p>
     * This is an in-place operation: the input tensor is overwritten with the result.
     *
     * @param tensor the tensor to reduce (modified in-place)
     */
    public void allReduceSum(INDArray tensor) {
        allReduce(tensor, tensor, REDUCE_SUM);
    }

    /**
     * AllReduce with specified operation.
     *
     * @param sendBuf  send buffer
     * @param recvBuf  receive buffer (can be same as sendBuf for in-place)
     * @param reduceOp reduction operation (REDUCE_SUM, REDUCE_MAX, etc.)
     */
    public void allReduce(INDArray sendBuf, INDArray recvBuf, int reduceOp) {
        ensureOpen();
        Pointer sendPtr = sendBuf.data().addressPointer();
        Pointer recvPtr = recvBuf.data().addressPointer();
        int dtype = dataTypeToNative(sendBuf.dataType());

        int result = Nd4j.getNativeOps().ncclDoAllReduce(
                commHandle, sendPtr, recvPtr,
                sendBuf.length(), dtype, reduceOp, null);

        if (result != 0) {
            throw new RuntimeException("NCCL AllReduce failed with error code " + result);
        }
    }

    /**
     * AllGather: gather data from all ranks.
     * <p>
     * Each rank contributes {@code localData} and receives all ranks' data
     * concatenated along the first dimension.
     *
     * @param localData this rank's data shard
     * @return gathered array with first dimension multiplied by numRanks
     */
    public INDArray allGather(INDArray localData) {
        ensureOpen();

        // Build output shape: first dim * numRanks
        long[] localShape = localData.shape();
        long[] gatheredShape = localShape.clone();
        gatheredShape[0] = localShape[0] * numRanks;

        INDArray gathered = Nd4j.create(localData.dataType(), gatheredShape);

        Pointer sendPtr = localData.data().addressPointer();
        Pointer recvPtr = gathered.data().addressPointer();
        int dtype = dataTypeToNative(localData.dataType());

        int result = Nd4j.getNativeOps().ncclDoAllGather(
                commHandle, sendPtr, recvPtr,
                localData.length(), dtype, null);

        if (result != 0) {
            gathered.close();
            throw new RuntimeException("NCCL AllGather failed with error code " + result);
        }

        return gathered;
    }

    /**
     * AllGather along a specific dimension (not the first).
     * <p>
     * Gathers along the last dimension, which is the typical pattern
     * for column-parallel linear output gathering.
     *
     * @param localData this rank's data shard
     * @param dim       dimension to gather along
     * @return gathered array
     */
    public INDArray allGatherAlongDim(INDArray localData, int dim) {
        ensureOpen();

        if (dim < 0) dim += localData.rank();

        long[] localShape = localData.shape();
        long[] gatheredShape = localShape.clone();
        gatheredShape[dim] = localShape[dim] * numRanks;

        INDArray gathered = Nd4j.create(localData.dataType(), gatheredShape);

        // NCCL AllGather always concatenates contiguously, so we need to
        // do the gather then reshape/transpose if dim != 0.
        // For dim=0 (batch) or dim=-1 (features), this is the common case.
        if (dim == 0) {
            Pointer sendPtr = localData.data().addressPointer();
            Pointer recvPtr = gathered.data().addressPointer();
            int dtype = dataTypeToNative(localData.dataType());

            int result = Nd4j.getNativeOps().ncclDoAllGather(
                    commHandle, sendPtr, recvPtr,
                    localData.length(), dtype, null);

            if (result != 0) {
                gathered.close();
                throw new RuntimeException("NCCL AllGather failed with error code " + result);
            }
        } else {
            // For non-zero dims, gather along dim 0 then transpose
            // This is a simplification; for production use, custom kernels would be better
            INDArray tempGathered = allGather(localData);

            // Reshape from [numRanks * batch, ...] to [numRanks, batch, ...]
            // then permute to put the gathered dim in the right place
            // For the common case of dim=-1 (last), we can use a simpler approach
            long elementsPerRank = localData.length();
            for (int r = 0; r < numRanks; r++) {
                // Copy each rank's contribution to the right position
                INDArray slice = tempGathered.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.interval(
                                r * localShape[0], (r + 1) * localShape[0]),
                        org.nd4j.linalg.indexing.NDArrayIndex.all());

                // Place into gathered at the appropriate offset along dim
                long offset = r * localShape[dim];
                if (dim == localData.rank() - 1 && localData.rank() == 2) {
                    gathered.get(
                            org.nd4j.linalg.indexing.NDArrayIndex.all(),
                            org.nd4j.linalg.indexing.NDArrayIndex.interval(offset, offset + localShape[dim])
                    ).assign(slice);
                }
            }
            tempGathered.close();
        }

        return gathered;
    }

    /**
     * ReduceScatter: reduce then scatter across ranks.
     *
     * @param sendBuf  full data to reduce and scatter
     * @param recvBuf  this rank's shard of the reduced result
     * @param reduceOp reduction operation
     */
    public void reduceScatter(INDArray sendBuf, INDArray recvBuf, int reduceOp) {
        ensureOpen();
        Pointer sendPtr = sendBuf.data().addressPointer();
        Pointer recvPtr = recvBuf.data().addressPointer();
        int dtype = dataTypeToNative(sendBuf.dataType());

        int result = Nd4j.getNativeOps().ncclDoReduceScatter(
                commHandle, sendPtr, recvPtr,
                recvBuf.length(), dtype, reduceOp, null);

        if (result != 0) {
            throw new RuntimeException("NCCL ReduceScatter failed with error code " + result);
        }
    }

    /**
     * Point-to-point send to a peer rank.
     *
     * @param data     data to send
     * @param peerRank destination rank
     */
    public void send(INDArray data, int peerRank) {
        ensureOpen();
        Pointer sendPtr = data.data().addressPointer();
        int dtype = dataTypeToNative(data.dataType());

        int result = Nd4j.getNativeOps().ncclDoSend(
                commHandle, sendPtr, data.length(),
                dtype, peerRank, null);

        if (result != 0) {
            throw new RuntimeException("NCCL Send to rank " + peerRank + " failed with error code " + result);
        }
    }

    /**
     * Point-to-point receive from a peer rank.
     *
     * @param recvBuf  buffer to receive into
     * @param peerRank source rank
     */
    public void recv(INDArray recvBuf, int peerRank) {
        ensureOpen();
        Pointer recvPtr = recvBuf.data().addressPointer();
        int dtype = dataTypeToNative(recvBuf.dataType());

        int result = Nd4j.getNativeOps().ncclDoRecv(
                commHandle, recvPtr, recvBuf.length(),
                dtype, peerRank, null);

        if (result != 0) {
            throw new RuntimeException("NCCL Recv from rank " + peerRank + " failed with error code " + result);
        }
    }

    /**
     * Begin a group of NCCL operations for fusion.
     * All operations between groupStart() and groupEnd() are launched
     * as a single fused NCCL operation.
     */
    public static void groupStart() {
        int result = Nd4j.getNativeOps().ncclGroupStart();
        if (result != 0) {
            throw new RuntimeException("NCCL groupStart failed");
        }
    }

    /**
     * End a group of NCCL operations.
     */
    public static void groupEnd() {
        int result = Nd4j.getNativeOps().ncclGroupEnd();
        if (result != 0) {
            throw new RuntimeException("NCCL groupEnd failed");
        }
    }

    /**
     * Check if NCCL is available in the current build.
     */
    public static boolean isAvailable() {
        try {
            Pointer p = Nd4j.getNativeOps().ncclGetUniqueId();
            if (p != null) {
                // Successfully created a unique ID - NCCL is available
                return true;
            }
            return false;
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public String backendName() {
        return "NCCL";
    }

    private void ensureOpen() {
        if (closed || commHandle == null) {
            throw new IllegalStateException("NcclCommunicator has been closed");
        }
    }

    private static int dataTypeToNative(DataType dt) {
        switch (dt) {
            case HALF: return 1;
            case FLOAT: return 5;
            case DOUBLE: return 7;
            case INT: return 3;
            case LONG: return 9;
            case BYTE: return 2;
            case UBYTE: return 6;
            case BFLOAT16: return 16;
            default:
                throw new IllegalArgumentException("Unsupported data type for NCCL: " + dt);
        }
    }

    @Override
    public void close() {
        if (!closed && commHandle != null) {
            Nd4j.getNativeOps().ncclCommDestroy(commHandle);
            commHandle = null;
            closed = true;
            log.info("NCCL communicator closed for rank {}", rank);
        }
    }

    @Override
    public String toString() {
        return "NcclCommunicator{rank=" + rank + "/" + numRanks +
                ", device=" + deviceId +
                ", open=" + !closed + '}';
    }
}
