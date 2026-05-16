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

package org.nd4j.linalg.jcublas.buffer;

import lombok.Getter;
import lombok.NonNull;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.nd4j.jita.allocator.enums.CudaConstants;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.impl.CudaDeallocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.HybridDataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.MultiBackendWorkspace;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.memory.MemcpyDirection;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.util.LongUtils;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.*;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Base class for a data buffer
 *
 * CUDA implementation for DataBuffer always uses JavaCPP
 * as allocationMode, and device access is masked by
 * appropriate allocator mover implementation.
 *
 * Memory allocation/deallocation is strictly handled by allocator,
 * since JavaCPP alloc/dealloc has nothing to do with CUDA.
 * But besides that, host pointers obtained from CUDA are 100%
 * compatible with CPU
 *
 * @author Adam Gibson
 * @author raver119@gmail.com
 */
public abstract class BaseCudaDataBuffer extends BaseDataBuffer implements JCudaBuffer, Deallocatable, HybridDataBuffer {

    private static final Logger log = LoggerFactory.getLogger(BaseCudaDataBuffer.class);

    @Getter
    protected transient volatile AllocationPoint allocationPoint;
    public final static long BASE_CUDA_DATA_BUFFER_OFFSET = RandomUtils.nextLong();

    private static AtomicAllocator allocator = AtomicAllocator.getInstance();


    protected DataType globalType = DataTypeUtil.getDtypeFromContext();

    // HybridDataBuffer device tracking fields
    protected volatile DeviceDescriptor ownerDevice;
    protected volatile DeviceDescriptor pinnedDevice;
    protected volatile boolean hostDirty = false;
    protected volatile boolean deviceDirty = false;
    protected volatile boolean cpuValid = false;   // CPU has valid copy of data
    protected volatile boolean gpuValid = true;    // GPU has valid copy (true by default for CUDA buffers)
    protected volatile MultiBackendWorkspace multiBackendWorkspace;

    public BaseCudaDataBuffer() {

    }


    public BaseCudaDataBuffer(long length, int elementSize, boolean initialize, @NonNull MemoryWorkspace workspace) {
        initTypeAndSize();
        initPointers(length, elementSize, initialize, workspace);
    }

    public BaseCudaDataBuffer(double[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocationMode.MIXED_DATA_TYPES;
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        initPointers(length, Nd4j.sizeOfDataType(dataType()), false, workspace);

        if (copy) {
            set(data, data.length, 0, 0);
        }
    }

    public BaseCudaDataBuffer(int[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocationMode.MIXED_DATA_TYPES;
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        initPointers(length, Nd4j.sizeOfDataType(dataType()), false, workspace);

        if (copy) {
            set(data, data.length, 0, 0);
        }
    }


    public OpaqueDataBuffer getOpaqueDataBuffer() {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        return ptrDataBuffer;
    }

    public BaseCudaDataBuffer(@NonNull Pointer pointer, @NonNull Pointer specialPointer, @NonNull Indexer indexer, long length) {
        this.allocationMode = AllocationMode.MIXED_DATA_TYPES;

        this.indexer = indexer;

        this.underlyingLength = length;
        this.length = length;

        initTypeAndSize();

        ptrDataBuffer = OpaqueDataBuffer.externalizedDataBuffer(length, this.type,  pointer, specialPointer);
        this.allocationPoint = new AllocationPoint(ptrDataBuffer, this.type.width() * length);

        this.deallocationId = ptrDataBuffer.getDeallocator() != null ?
            ptrDataBuffer.getDeallocator().getUniqueId() : -1;
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");
    }

    /**
     * Meant for creating another view of a buffer
     *
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length  the length of the view
     */
    public BaseCudaDataBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);

        // allocating interop buffer
        this.ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(length, type, true);
        // passing existing pointer to native holder
        this.ptrDataBuffer.setPrimaryBuffer(pointer, length);

        //cuda specific bits
        this.allocationPoint = new AllocationPoint(ptrDataBuffer, length * elementSize);
        allocationPoint.tickHostWrite();
        this.deallocationId = ptrDataBuffer.getDeallocator() != null ?
            ptrDataBuffer.getDeallocator().getUniqueId() : -1;

        // now we're getting context and copying our stuff to device
        val context = AtomicAllocator.getInstance().getDeviceContext();

        val perfD = PerformanceTracker.getInstance().helperStartTransaction();

        NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(allocationPoint.getDevicePointer(), pointer, length * getElementSize(), CudaConstants.cudaMemcpyHostToDevice, context.getSpecialStream());

        PerformanceTracker.getInstance().helperRegisterTransaction(allocationPoint.getDeviceId(), perfD / 2, allocationPoint.getNumberOfBytes(), MemcpyDirection.HOST_TO_DEVICE);
        context.getSpecialStream().synchronize();
    }

    protected void initPointers(long length, DataType dtype, boolean initialize) {
        initPointers(length, Nd4j.sizeOfDataType(dtype), initialize);
    }

    public void lazyAllocateHostPointer() {
        if (length() == 0)
            return;

        // java side might be unaware of native-side buffer allocation
        if (this.indexer == null || this.pointer == null || this.pointer.address() == 0) {
            initHostPointerAndIndexer();
        } else if (allocationPoint.getHostPointer() != null && allocationPoint.getHostPointer().address() != this.pointer.address()) {
            initHostPointerAndIndexer();
        }
    }

    protected BaseCudaDataBuffer(ByteBuffer buffer, DataType dtype, long length) {
        this(buffer, dtype, length, 0L, false);
    }

    /**
     * Constructor for ByteBuffer with optional CPU-only allocation.
     * When cpuOnly is true, the data is loaded to CPU memory only and GPU
     * allocation is deferred until the data is actually needed on the device.
     * This enables faster model loading by avoiding immediate GPU transfers.
     *
     * @param buffer the source ByteBuffer
     * @param dtype the data type
     * @param length number of elements
     * @param cpuOnly if true, allocate only on CPU (defer GPU allocation)
     */
    protected BaseCudaDataBuffer(ByteBuffer buffer, DataType dtype, long length, boolean cpuOnly) {
        this(buffer, dtype, length, 0L, cpuOnly);
    }

    protected BaseCudaDataBuffer(ByteBuffer buffer, DataType dtype, long length, long offset) {
        this(buffer, dtype, length, offset, false);
    }

    /**
     * Constructor for ByteBuffer with offset and optional CPU-only allocation.
     *
     * @param buffer the source ByteBuffer
     * @param dtype the data type
     * @param length number of elements
     * @param offset offset in elements
     * @param cpuOnly if true, allocate only on CPU (defer GPU allocation)
     */
    protected BaseCudaDataBuffer(ByteBuffer buffer, DataType dtype, long length, long offset, boolean cpuOnly) {
        this(length, Nd4j.sizeOfDataType(dtype), false, cpuOnly);

        Pointer temp = null;

        switch (dataType()) {
            case DOUBLE:
                temp = new DoublePointer(buffer.asDoubleBuffer());
                break;
            case FLOAT:
                temp = new FloatPointer(buffer.asFloatBuffer());
                break;
            case HALF:
                temp = new ShortPointer(buffer.asShortBuffer());
                break;
            case LONG:
                temp = new LongPointer(buffer.asLongBuffer());
                break;
            case INT:
                temp = new IntPointer(buffer.asIntBuffer());
                break;
            case SHORT:
                temp = new ShortPointer(buffer.asShortBuffer());
                break;
            case UBYTE: //Fall through
            case BYTE:
                temp = new BytePointer(buffer);
                break;
            case BOOL:
                temp = new BooleanPointer(length());
                break;
            case UTF8:
                temp = new BytePointer(length());
                break;
            case BFLOAT16:
                temp = new ShortPointer(length());
                break;
            case UINT16:
                temp = new ShortPointer(length());
                break;
            case UINT32:
                temp = new IntPointer(length());
                break;
            case UINT64:
                temp = new LongPointer(length());
                break;
        }

        if (offset > 0)
            temp = new PagedPointer(temp.address() + offset * getElementSize());

        if (cpuOnly) {
            // CPU-only mode: copy data to host buffer only, skip GPU transfer
            val hostPtr = allocationPoint.getHostPointer();
            if (hostPtr != null && hostPtr.address() != 0) {
                Pointer.memcpy(hostPtr, temp, length * Nd4j.sizeOfDataType(dtype));
            }
            // Mark host as valid, device as needing update
            allocationPoint.tickHostWrite();
            this.cpuValid = true;
            this.gpuValid = false;
            log.debug("Loaded {} elements to CPU-only buffer (lazy GPU migration)", length);
        } else {
            // Normal mode: copy data to device
            val stream = AtomicAllocator.getInstance().getDeviceContext().getSpecialStream();
            val ptr = ptrDataBuffer.specialBuffer();

            NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(ptr, temp, length * Nd4j.sizeOfDataType(dtype), CudaConstants.cudaMemcpyHostToDevice, stream);
            stream.synchronize();

            // mark device buffer as updated
            allocationPoint.tickDeviceWrite();
        }
    }

    @Override
    public boolean shouldDeAllocate() {
        return !released.get() && !isConstant();
    }

    protected void initHostPointerAndIndexer() {
        if (length() == 0)
            return;

        if (allocationPoint.getHostPointer() == null) {
            val location = allocationPoint.getAllocationStatus();
            if (parentWorkspace == null) {
                // let cpp allocate primary buffer
                NativeOpsHolder.getInstance().getDeviceNativeOps().dbAllocatePrimaryBuffer(ptrDataBuffer);
            } else {
                val ptr = parentWorkspace.alloc(this.length * this.elementSize, MemoryKind.HOST, this.dataType(), false);
                ptrDataBuffer.setPrimaryBuffer(ptr, this.length);
            }
            this.allocationPoint.setAllocationStatus(location);
            this.allocationPoint.tickDeviceWrite();
        }

        val hostPointer = allocationPoint.getHostPointer();

        assert hostPointer != null;

        initPointerAndIndexerFromHost(hostPointer);
    }

    private void initPointerAndIndexerFromHost(Pointer hostPointer) {
        switch (dataType()) {
            case DOUBLE:
                this.pointer = new CudaPointer(hostPointer, length, 0).asDoublePointer();
                indexer = DoubleIndexer.create((DoublePointer) pointer);
                break;
            case FLOAT:
                this.pointer = new CudaPointer(hostPointer, length, 0).asFloatPointer();
                indexer = FloatIndexer.create((FloatPointer) pointer);
                break;
            case UINT32:
                this.pointer = new CudaPointer(hostPointer, length, 0).asIntPointer();
                indexer = UIntIndexer.create((IntPointer) pointer);
                break;
            case INT:
                this.pointer = new CudaPointer(hostPointer, length, 0).asIntPointer();
                indexer = IntIndexer.create((IntPointer) pointer);
                break;
            case BFLOAT16:
                this.pointer = new CudaPointer(hostPointer, length, 0).asShortPointer();
                indexer = Bfloat16Indexer.create((ShortPointer) pointer);
                break;
            case HALF:
                this.pointer = new CudaPointer(hostPointer, length, 0).asShortPointer();
                indexer = HalfIndexer.create((ShortPointer) pointer);
                break;
            case UINT64:    //Fall through
                this.pointer = new CudaPointer(hostPointer, length, 0).asLongPointer();
                indexer = ULongIndexer.create((LongPointer) pointer);
                break;
            case LONG:
                this.pointer = new CudaPointer(hostPointer, length, 0).asLongPointer();
                indexer = LongIndexer.create((LongPointer) pointer);
                break;
            case UINT16:
                this.pointer = new CudaPointer(hostPointer, length, 0).asShortPointer();
                indexer = UShortIndexer.create((ShortPointer) pointer);
                break;
            case SHORT:
                this.pointer = new CudaPointer(hostPointer, length, 0).asShortPointer();
                indexer = ShortIndexer.create((ShortPointer) pointer);
                break;
            case UBYTE:
                this.pointer = new CudaPointer(hostPointer, length, 0).asBytePointer();
                indexer = UByteIndexer.create((BytePointer) pointer);
                break;
            case BYTE:
                this.pointer = new CudaPointer(hostPointer, length, 0).asBytePointer();
                indexer = ByteIndexer.create((BytePointer) pointer);
                break;
            case BOOL:
                this.pointer = new CudaPointer(hostPointer, length, 0).asBooleanPointer();
                indexer = BooleanIndexer.create((BooleanPointer) pointer);
                break;
            case UTF8:
                this.pointer = new CudaPointer(hostPointer, length, 0).asBytePointer();
                indexer = ByteIndexer.create((BytePointer) pointer);
                break;
            default:
                throw new UnsupportedOperationException();
        }
    }

    protected void initPointers(long length, int elementSize, boolean initialize) {
        this.allocationMode = AllocationMode.MIXED_DATA_TYPES;
        this.length = length;
        this.elementSize =  (byte) elementSize;

        // Allocate both host and device buffers. The native C++ layer (CudaMemoryPool)
        // handles device selection and OOM failover (trim pool + retry, other GPUs, pinned host).
        ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(length, type, true);
        this.allocationPoint = new AllocationPoint(ptrDataBuffer, length * type.width());

        if (initialize) {
            val devicePtr = allocationPoint.getDevicePointer();
            int bufferDeviceId = allocationPoint.getDeviceId();
            val nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            int currentDevice = nativeOps.getDevice();

            try {
                // Switch to the buffer's device if needed for the memset operation
                if (bufferDeviceId >= 0 && bufferDeviceId != currentDevice) {
                    nativeOps.setDevice(bufferDeviceId);
                }

                // Use synchronous memset - no stream needed
                int result = nativeOps.memsetSync(devicePtr, 0, length * elementSize, 0, null);
                if (result == 0) {
                    throw new RuntimeException("memsetSync failed in initPointers");
                }
            } finally {
                // Switch back to original device
                if (bufferDeviceId >= 0 && bufferDeviceId != currentDevice) {
                    nativeOps.setDevice(currentDevice);
                    AtomicAllocator.getInstance().getMemoryHandler().resetCachedContext();
                }
            }
        }

        this.deallocationId = ptrDataBuffer.getDeallocator() != null ?
            ptrDataBuffer.getDeallocator().getUniqueId() : -1;
    }

    /**
     * Initialize pointers with optional CPU-only allocation for lazy GPU migration.
     * When cpuOnly is true, only the host buffer is allocated. GPU allocation
     * is deferred until the data is actually needed on the device through
     * ensureAvailableOn() or similar methods.
     *
     * @param length number of elements
     * @param elementSize size of each element in bytes
     * @param initialize if true, zero-initialize the buffer
     * @param cpuOnly if true, allocate only on CPU (defer GPU allocation)
     */
    protected void initPointersCpuOnly(long length, int elementSize, boolean initialize, boolean cpuOnly) {
        if (!cpuOnly) {
            // Fall back to normal allocation
            initPointers(length, elementSize, initialize);
            return;
        }

        // CPU-only allocation for lazy GPU migration
        this.allocationMode = AllocationMode.MIXED_DATA_TYPES;
        initTypeAndSize();

        this.length = length;
        this.underlyingLength = length;

        // Allocate both host and device memory so we can copy data to host
        ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(length, type, true);
        this.allocationPoint = new AllocationPoint(ptrDataBuffer, length * elementSize);

        // Mark CPU as valid, GPU as invalid (deferred)
        this.gpuValid = false;
        this.cpuValid = true;

        this.deallocationId = ptrDataBuffer.getDeallocator() != null ?
            ptrDataBuffer.getDeallocator().getUniqueId() : -1;

        log.debug("Created CPU-only buffer for lazy GPU migration: {} elements, {} bytes", length, length * elementSize);
    }

    protected void initPointers(long length, int elementSize, boolean initialize, @NonNull MemoryWorkspace workspace) {
        this.allocationMode = AllocationMode.MIXED_DATA_TYPES;
        initTypeAndSize();

        this.attached = true;
        this.parentWorkspace = workspace;

        this.length = length;

        // Use synchronous memset to avoid stream/device mismatch issues in multi-GPU setups
        val nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        if (workspace.getWorkspaceConfiguration().getPolicyMirroring() == MirroringPolicy.FULL) {
            val devicePtr = workspace.alloc(length * elementSize, MemoryKind.DEVICE, type, initialize);

            // allocate from workspace, and pass it  to native DataBuffer
            ptrDataBuffer = OpaqueDataBuffer.externalizedDataBuffer(this.length, type, null, devicePtr);

            if (initialize) {
                int result = nativeOps.memsetSync(devicePtr, 0, length * elementSize, 0, null);
                if (result == 0) {
                    throw new RuntimeException("memsetSync failed in initPointers (workspace FULL mirroring)");
                }
            }
        }  else {
            // Non-FULL mirroring: allocate based on actual policy.
            // HOST policy: allocate pinned host memory, pass as primary (host pointer).
            // DEVICE policy: allocate device memory, pass as special (device pointer).
            val mirroringPolicy = workspace.getWorkspaceConfiguration().getPolicyMirroring();
            if (mirroringPolicy == MirroringPolicy.HOST_ONLY) {
                val hostPtr = workspace.alloc(length * elementSize, MemoryKind.HOST, type, initialize);
                ptrDataBuffer = OpaqueDataBuffer.externalizedDataBuffer(this.length, type, hostPtr, null);

                if (initialize) {
                    int result = nativeOps.memsetSync(hostPtr, 0, length * elementSize, 0, null);
                    if (result == 0) {
                        throw new RuntimeException("memsetSync failed in initPointers (workspace HOST)");
                    }
                }
            } else {
                // DEVICE or default: allocate device memory, pass as special (device pointer)
                val devicePtr = workspace.alloc(length * elementSize, MemoryKind.DEVICE, type, initialize);
                ptrDataBuffer = OpaqueDataBuffer.externalizedDataBuffer(this.length, type, null, devicePtr);

                if (initialize) {
                    int result = nativeOps.memsetSync(devicePtr, 0, length * elementSize, 0, null);
                    if (result == 0) {
                        throw new RuntimeException("memsetSync failed in initPointers (workspace DEVICE)");
                    }
                }
            }
        }

        this.allocationPoint = new AllocationPoint(ptrDataBuffer, elementSize * length);

        this.deallocationId = ptrDataBuffer.getDeallocator() != null ?
            ptrDataBuffer.getDeallocator().getUniqueId() : -1;

        workspaceGenerationId = workspace.getGenerationId();
        this.attached = true;
        this.parentWorkspace = workspace;
        initHostPointerAndIndexer();
    }

    /**
     * Initialize data type and element size
     */
    protected void initTypeAndSize() {
        if (dataType() == null) {
            throw new IllegalStateException("No data type specified.");
        }
        elementSize = (byte) Nd4j.sizeOfDataType(dataType());
    }

    /**
     * Base constructor. It's used within all constructors internally
     *
     * @param length      the length of the buffer
     * @param elementSize the size of each element
     */
    public BaseCudaDataBuffer(long length, int elementSize) {
        this(length, elementSize, true);
    }

    public BaseCudaDataBuffer(long length, int elementSize, MemoryWorkspace workspace) {
        this(length, elementSize, true, workspace);
    }

    public BaseCudaDataBuffer(long length, int elementSize, boolean initialize) {
        initTypeAndSize();
        initPointers(length, elementSize, initialize);
    }

    /**
     * Constructor with optional CPU-only allocation for lazy GPU migration.
     * When cpuOnly is true, only the host buffer is allocated and GPU allocation
     * is deferred until the data is actually needed on the device.
     *
     * @param length number of elements
     * @param elementSize size of each element in bytes
     * @param initialize if true, zero-initialize the buffer
     * @param cpuOnly if true, allocate only on CPU (defer GPU allocation)
     */
    public BaseCudaDataBuffer(long length, int elementSize, boolean initialize, boolean cpuOnly) {
        initTypeAndSize();
        initPointersCpuOnly(length, elementSize, initialize, cpuOnly);
    }

    public BaseCudaDataBuffer(long length, boolean initialize) {
        if (length < 0)
            throw new IllegalArgumentException("Length must be >= 0");
        initTypeAndSize();
        this.length = length;
        this.underlyingLength = length;
        allocationMode = AllocationMode.MIXED_DATA_TYPES;
        if (length < 0)
            throw new IllegalArgumentException("Unable to create a buffer of length <= 0");

        initPointers(length, elementSize, initialize);
    }

    public BaseCudaDataBuffer(long length, boolean initialize, @NonNull MemoryWorkspace workspace) {
        if (length < 0)
            throw new IllegalArgumentException("Length must be >= 0");
        initTypeAndSize();
        this.length = length;
        this.underlyingLength = length;
        allocationMode = AllocationMode.MIXED_DATA_TYPES;
        if (length < 0)
            throw new IllegalArgumentException("Unable to create a buffer of length <= 0");

        this.attached = true;
        this.parentWorkspace = workspace;

        initPointers(length, elementSize, initialize, workspace);
    }

    public BaseCudaDataBuffer(long length) {
        this(length, Nd4j.sizeOfDataType(Nd4j.dataType()));
    }

    public BaseCudaDataBuffer(float[] data) {
        this(data, true);
    }

    public BaseCudaDataBuffer(float[] data, boolean copy) {
        allocationMode = AllocationMode.MIXED_DATA_TYPES;
        initTypeAndSize();

        this.length = data.length;
        this.underlyingLength = data.length;

        initPointers(length, Nd4j.sizeOfDataType(dataType()), false);

        if (copy) {
            set(data, data.length, 0, 0);
        }
    }

    public BaseCudaDataBuffer(float[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocationMode.MIXED_DATA_TYPES;
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        initPointers(length, Nd4j.sizeOfDataType(dataType()), false, workspace);

        if (copy) {
            set(data, data.length, 0, 0);
        }
    }

    public BaseCudaDataBuffer(float[] data, MemoryWorkspace workspace) {
        this(data, true, workspace);
    }




    /**
     * This method always returns host pointer
     *
     * @return
     */
    @Override
    public long address() {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        if(allocationPoint.getHostPointer() == null)
            return -1;
        return allocationPoint.getHostPointer().address();
    }

    @Override
    public long platformAddress() {
        return allocationPoint.getDevicePointer().address();
    }

    @Override
    public Pointer pointer() {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        lazyAllocateHostPointer();

        return super.pointer();
    }



    public void copyDataFromSrc(Pointer pointer, long length, long srcOffset, long dstOffset) {
        val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));
        val nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        val context = AtomicAllocator.getInstance().getDeviceContext();

        // Do NOT call setPrimaryBuffer with the temporary pointer!
        // The temporary pointer (e.g., LongPointer wrapping a Java array) will be freed
        // when it goes out of scope, causing use-after-free when syncToPrimary is called.

        // Instead, ensure we have properly allocated HOST memory first
        if (allocationPoint.getHostPointer() == null || allocationPoint.getHostPointer().isNull()) {
            nativeOps.dbAllocatePrimaryBuffer(ptrDataBuffer);
        }

        // Get the properly allocated HOST pointer
        Pointer hostPtr = allocationPoint.getHostPointer();
        if (hostPtr == null || hostPtr.isNull()) {
            throw new IllegalStateException("Failed to allocate HOST buffer for data copy");
        }

        // Copy from source Java array to our persistent HOST buffer
        val dstHostPtr = new CudaPointer(hostPtr.address() + (dstOffset * elementSize));
        Pointer.memcpy(dstHostPtr, srcPtr, length * getElementSize());

        // Copy from HOST buffer to DEVICE (if device buffer exists).
        // For HOST_ONLY workspace buffers, there is no device buffer — data lives only on host.
        Pointer devPtr = allocationPoint.getDevicePointer();
        if (devPtr != null && !devPtr.isNull()) {
            val perfD = PerformanceTracker.getInstance().helperStartTransaction();
            val dstDevPtr = new CudaPointer(devPtr.address() + (dstOffset * elementSize));

            // Use synchronous memcpy to avoid stream/device mismatch issues in multi-GPU setups
            int result = nativeOps.memcpySync(
                    dstDevPtr,
                    dstHostPtr,
                    length * getElementSize(),
                    CudaConstants.cudaMemcpyHostToDevice,
                    null);  // null stream means synchronous

            if (result == 0) {
                throw new RuntimeException("memcpySync failed in copyDataFromSrc");
            }

            PerformanceTracker.getInstance().helperRegisterTransaction(
                    allocationPoint.getDeviceId(), perfD / 2,
                    allocationPoint.getNumberOfBytes(), MemcpyDirection.HOST_TO_DEVICE);

            allocationPoint.tickDeviceWrite();
            nativeOps.dbTickDeviceWrite(ptrDataBuffer);
        }

        // Mark HOST as having valid data on BOTH Java and C++ sides.
        // Without dbTickHostWrite, the C++ DataBuffer doesn't know the host was written,
        // so isPrimaryActual() returns false. If syncToPrimary() is later called and the
        // device buffer happens to be stale/zero, it would overwrite the correct host data
        // with device zeros. By ticking host write on C++, isPrimaryActual() returns true
        // and syncToPrimary() becomes a no-op, preserving the correct host data.
        allocationPoint.tickHostWrite();
        nativeOps.dbTickHostWrite(ptrDataBuffer);
    }

    /**
     *
     * PLEASE NOTE: length, srcOffset, dstOffset are considered numbers of elements, not byte offsets
     *
     * @param data
     * @param length
     * @param srcOffset
     * @param dstOffset
     */
    public void set(int[] data, long length, long srcOffset, long dstOffset) {
        // TODO: make sure getPointer returns proper pointer
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        switch (dataType()) {
            case BOOL: {
                val pointer = new BytePointer(ArrayUtil.toBytes(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);
            }
            break;

            //note we drop down on purpose here and use bytes copy data
            case UBYTE: {
                data = ArrayUtil.cutBelowZero(data);
            }
            case BYTE: {
                val pointer = new BytePointer(ArrayUtil.toBytes(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            //drop down on purpose
            case UINT16:
                data = ArrayUtil.cutBelowZero(data);
            case SHORT: {
                val pointer = new ShortPointer(ArrayUtil.toShorts(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            //drop down on purpose
            case UINT32:
                data = ArrayUtil.cutBelowZero(data);

            case INT: {
                val pointer = new IntPointer(data);
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;

            case BFLOAT16: {
                val pointer = new ShortPointer(ArrayUtil.toBfloats(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            //drop down on purpose
            case UINT64:
                data = ArrayUtil.cutBelowZero(data);
            case LONG: {
                val pointer = new LongPointer(LongUtils.toLongs(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case HALF: {
                val pointer = new ShortPointer(ArrayUtil.toHalfs(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case FLOAT: {
                val pointer = new FloatPointer(ArrayUtil.toFloats(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case DOUBLE: {
                val pointer = new DoublePointer(ArrayUtil.toDouble(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }



    }



    public void set(long[] data, long length, long srcOffset, long dstOffset) {
        // TODO: make sure getPointer returns proper pointer
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        switch (dataType()) {
            case BOOL: {
                val pointer = new BytePointer(ArrayUtil.toBytes(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);


            }
            break;


            //drop down on purpose and use the bytes conversion
            case UBYTE: {
                data = ArrayUtil.cutBelowZero(data);

            }

            case BYTE: {
                val pointer = new BytePointer(ArrayUtil.toBytes(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);


            }
            break;
            case UINT16: {
                data = ArrayUtil.cutBelowZero(data);
                val pointer = new ShortPointer(ArrayUtil.toShorts(data));
                copyDataFromSrc(pointer, length, srcOffset, dstOffset);
            }
            break;
            case SHORT: {
                val pointer = new ShortPointer(ArrayUtil.toShorts(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);
            }
            break;
            case UINT32:
                data = ArrayUtil.cutBelowZero(data);
            case INT: {
                val pointer = new IntPointer(ArrayUtil.toInts(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case UINT64:
                data = ArrayUtil.cutBelowZero(data);
            case LONG: {
                val pointer = new LongPointer(data);
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);
            }
            break;
            case BFLOAT16: {

                val pointer = new ShortPointer(ArrayUtil.toBfloats(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);
            }
            break;
            case HALF: {
                val pointer = new ShortPointer(ArrayUtil.toHalfs(data));
                // we're keeping pointer reference for JVM
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);
            }
            break;
            case FLOAT: {
                val pointer = new FloatPointer(ArrayUtil.toFloats(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);


            }
            break;
            case DOUBLE: {
                val pointer = new DoublePointer(ArrayUtil.toDouble(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);
            }
            break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }



    }

    /**
     *
     * PLEASE NOTE: length, srcOffset, dstOffset are considered numbers of elements, not byte offsets
     *
     * @param data
     * @param length
     * @param srcOffset
     * @param dstOffset
     */
    public void set(float[] data, long length, long srcOffset, long dstOffset) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        switch (dataType()) {
            case BOOL: {
                BytePointer pointer = new BytePointer(ArrayUtil.toBytes(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;

            case BFLOAT16: {
                ShortPointer pointer = new ShortPointer(ArrayUtil.toBfloats(data));
                copyDataFromSrc(pointer, length, srcOffset, dstOffset);
            }
            break;
            case UBYTE: {
                data = ArrayUtil.cutBelowZero(data);

            }
            case BYTE: {
                BytePointer pointer = new BytePointer(ArrayUtil.toBytes(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            //drop down on purpose
            case UINT16:
                data = ArrayUtil.cutBelowZero(data);
            case SHORT: {
                val pointer = new ShortPointer(ArrayUtil.toShorts(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            //drop down on purpose
            case UINT32:
                data = ArrayUtil.cutBelowZero(data);
            case INT: {
                IntPointer pointer = new IntPointer(ArrayUtil.toInts(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            //drop down on purpose
            case UINT64:
                data = ArrayUtil.cutBelowZero(data);
            case LONG: {
                LongPointer pointer = new LongPointer(ArrayUtil.toLongArray(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case HALF: {
                ShortPointer pointer = new ShortPointer(ArrayUtil.toHalfs(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case FLOAT: {
                FloatPointer pointer = new FloatPointer(data);
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case DOUBLE: {
                DoublePointer pointer = new DoublePointer(ArrayUtil.toDoubles(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }



    }



    /**
     *
     * PLEASE NOTE: length, srcOffset, dstOffset are considered numbers of elements, not byte offsets
     *
     * @param data
     * @param length
     * @param srcOffset
     * @param dstOffset
     */
    public void set(short[] data, long length, long srcOffset, long dstOffset) {
        switch (dataType()) {
            case BOOL:  {
                val pointer = new BytePointer(ArrayUtil.toBytes(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            //drop down on purpose, use the byte conversion routine below
            case UBYTE: {
                data = ArrayUtil.cutBelowZero(data);
            }
            case BYTE: {
                val pointer = new BytePointer(ArrayUtil.toBytes(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;

            //drop down on purpose
            case UINT16:
                data = ArrayUtil.cutBelowZero(data);
            case SHORT: {
                val pointer = new ShortPointer(data);
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            //drop down on purpose
            case UINT32:
                data = ArrayUtil.cutBelowZero(data);
            case INT: {
                val pointer = new IntPointer(ArrayUtil.toInts(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case UINT64:
                data = ArrayUtil.cutBelowZero(data);
            case LONG: {
                val pointer = new LongPointer(ArrayUtil.toLongs(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case HALF: {
                val pointer = new ShortPointer(ArrayUtil.toHalfs(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;

            case BFLOAT16: {
                val pointer = new ShortPointer(ArrayUtil.toBfloats(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case FLOAT: {
                val pointer = new FloatPointer(ArrayUtil.toFloats(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case DOUBLE: {
                val pointer = new DoublePointer(ArrayUtil.toDoubleArray(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }


    }

    /**
     *
     * PLEASE NOTE: length, srcOffset, dstOffset are considered numbers of elements, not byte offsets
     *
     * @param data
     * @param length
     * @param srcOffset
     * @param dstOffset
     */
    public void set(byte[] data, long length, long srcOffset, long dstOffset) {
        switch (dataType()) {
            case BOOL:  {
                val pointer = new BooleanPointer(ArrayUtil.toBooleanArray(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            //drop down on purpose, use the byte conversion routine below

            case UBYTE: {
                data = ArrayUtil.cutBelowZero(data);
            }
            case BYTE: {
                val pointer = new BytePointer(data);
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;

            case UINT16:
                data = ArrayUtil.cutBelowZero(data);

            case SHORT: {
                val pointer = new ShortPointer(ArrayUtil.toShorts(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;

            case BFLOAT16: {
                val pointer = new ShortPointer(ArrayUtil.toBfloats(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;

            case UINT32:
                data = ArrayUtil.cutBelowZero(data);
            case INT: {
                val pointer = new IntPointer(ArrayUtil.toInts(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case UINT64:
                data = ArrayUtil.cutBelowZero(data);
            case LONG: {
                val pointer = new LongPointer(ArrayUtil.toLongs(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case HALF: {
                val pointer = new ShortPointer(ArrayUtil.toHalfs(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case FLOAT: {
                val pointer = new FloatPointer(ArrayUtil.toFloats(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case DOUBLE: {
                val pointer = new DoublePointer(ArrayUtil.toDouble(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }


    }

    /**
     *
     * PLEASE NOTE: length, srcOffset, dstOffset are considered numbers of elements, not byte offsets
     *
     * @param data
     * @param length
     * @param srcOffset
     * @param dstOffset
     */
    public void set(boolean[] data, long length, long srcOffset, long dstOffset) {
        switch (dataType()) {
            case BOOL:  {
                val pointer = new BooleanPointer(data);
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;

            case UBYTE:
                //note this is on purpose. no data is below zero with booleans
            case BYTE: {
                val pointer = new BytePointer(ArrayUtil.toBytes(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case UINT16:
                //note this is on purpose. no data is below zero with booleans
            case SHORT: {
                val pointer = new ShortPointer(ArrayUtil.toShorts(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case UINT32:
                //note this is on purpose. no data is below zero with booleans
            case INT: {
                val pointer = new IntPointer(ArrayUtil.toInts(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case UINT64:
                //note this is on purpose. no data is below zero with booleans
            case LONG: {
                val pointer = new LongPointer(ArrayUtil.toLongs(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case HALF: {
                val pointer = new ShortPointer(ArrayUtil.toHalfs(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;

            case BFLOAT16: {
                val pointer = new ShortPointer(ArrayUtil.toBfloats(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case FLOAT: {
                val pointer = new FloatPointer(ArrayUtil.toFloats(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case DOUBLE: {
                val pointer = new DoublePointer(ArrayUtil.toDouble(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }


    }

    /**
     *
     * PLEASE NOTE: length, srcOffset, dstOffset are considered numbers of elements, not byte offsets
     *
     * @param data
     * @param length
     * @param srcOffset
     * @param dstOffset
     */
    public void set(double[] data, long length, long srcOffset, long dstOffset) {
        switch (dataType()) {
            case BOOL:  {
                val pointer = new BytePointer(ArrayUtil.toBytes(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;

            case UBYTE: {
                data = ArrayUtil.cutBelowZero(data);
            }
            case BYTE: {
                val pointer = new BytePointer(ArrayUtil.toBytes(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;

            case SHORT: {
                val pointer = new ShortPointer(ArrayUtil.toShorts(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case INT: {
                val pointer = new IntPointer(ArrayUtil.toInts(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case LONG: {
                val pointer = new LongPointer(ArrayUtil.toLongs(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case HALF: {
                val pointer = new ShortPointer(ArrayUtil.toHalfs(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;

            case BFLOAT16: {
                val pointer = new ShortPointer(ArrayUtil.toBfloats(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case FLOAT: {
                val pointer = new FloatPointer(ArrayUtil.toFloats(data));
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            case DOUBLE: {
                val pointer = new DoublePointer(data);
                copyDataFromSrc(pointer,length,srcOffset,dstOffset);

            }
            break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }


    }

    @Override
    public void setData(byte[] data) {
        if (data.length == 0)
            return;

        set(data, data.length, 0, 0);
    }

    @Override
    public void setData(short[] data) {
        if (data.length == 0)
            return;

        set(data, data.length, 0, 0);
    }

    @Override
    public void setData(boolean[] data) {
        if (data.length == 0)
            return;

        set(data, data.length, 0, 0);
    }

    @Override
    public void setData(int[] data) {
        if (data.length == 0)
            return;

        set(data, data.length, 0, 0);
    }

    @Override
    public void setData(long[] data) {
        if (data.length == 0)
            return;

        set(data, data.length, 0, 0);
    }

    @Override
    public void setData(float[] data) {
        if (data.length == 0)
            return;

        set(data, data.length, 0, 0);
    }

    @Override
    public void setData(double[] data) {
        if (data.length == 0)
            return;

        set(data, data.length, 0, 0);
    }

    @Override
    protected void setNioBuffer() {
        throw new UnsupportedOperationException("setNioBuffer() is not supported for CUDA backend");
    }

    @Override
    public void copyAtStride(DataBuffer buf, long n, long stride, long yStride, long offset, long yOffset) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        allocator.synchronizeHostData(buf);
        super.copyAtStride(buf, n, stride, yStride, offset, yOffset);
    }

    @Override
    public AllocationMode allocationMode() {
        return allocationMode;
    }

    @Override
    public ByteBuffer getHostBuffer() {
        return pointer.asByteBuffer();
    }

    @Override
    public Pointer getHostPointer() {
        return AtomicAllocator.getInstance().getHostPointer(this);
    }

    @Override
    public Pointer getHostPointer(long offset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void removeReferencing(String id) {
    }

    @Override
    public Collection<String> references() {
        return null;
    }

    @Override
    public int getElementSize() {
        return elementSize;
    }


    @Override
    public void addReferencing(String id) {
    }


    @Deprecated
    public Pointer getHostPointer(INDArray arr, int stride, long offset, int length) {
        throw new UnsupportedOperationException("This method is deprecated");
    }

    @Deprecated
    public void set(Pointer pointer) {
        throw new UnsupportedOperationException("set(Pointer) is not supported");
    }

    @Override
    public void put(long i, float element) {
        synchronized (this) {
            ensureCorrectDevice();
            lazyAllocateHostPointer();
            allocator.synchronizeHostData(this);
            allocator.tickHostWrite(this);
            super.put(i, element);
        }
    }

    /**
     * Ensures the current thread is on this buffer's device before performing operations.
     * This is critical for multi-device scenarios where threads may access buffers from different devices.
     */
    private void ensureCorrectDevice() {
        if (allocationPoint != null) {
            int bufferDevice = allocationPoint.getDeviceId();
            int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            if (bufferDevice >= 0 && bufferDevice != currentDevice) {
                DeviceMemoryManager.getInstance().switchDevice(bufferDevice, "BaseCudaDataBuffer.ensureCorrectDevice", "buffer-device-align");
            }
        }
    }

    @Override
    public void put(long i, boolean element) {
        synchronized (this) {
            ensureCorrectDevice();
            lazyAllocateHostPointer();
            allocator.synchronizeHostData(this);
            allocator.tickHostWrite(this);
            super.put(i, element);
        }
    }

    @Override
    public void put(long i, double element) {
        synchronized (this) {
            ensureCorrectDevice();
            lazyAllocateHostPointer();
            allocator.synchronizeHostData(this);
            allocator.tickHostWrite(this);
            super.put(i, element);
        }
    }

    @Override
    public void put(long i, short element) {
        synchronized (this) {
            ensureCorrectDevice();
            lazyAllocateHostPointer();
            allocator.synchronizeHostData(this);
            allocator.tickHostWrite(this);
            super.put(i, element);
        }
    }

    @Override
    public void put(long i, int element) {
        synchronized (this) {
            ensureCorrectDevice();
            lazyAllocateHostPointer();
            allocator.synchronizeHostData(this);
            allocator.tickHostWrite(this);
            super.put(i, element);
        }
    }

    @Override
    public void put(long i, long element) {
        synchronized (this) {
            ensureCorrectDevice();
            lazyAllocateHostPointer();
            allocator.synchronizeHostData(this);
            allocator.tickHostWrite(this);
            super.put(i, element);
        }
    }

    @Override
    public Pointer addressPointer() {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        return AtomicAllocator.getInstance().getHostPointer(this);
    }

    /**
     * Set an individual element
     *
     * @param index the index of the element
     * @param from  the element to get data from
     */
    @Deprecated
    protected void set(long index, long length, Pointer from, long inc) {
        long offset = getElementSize() * index;
        if (offset >= length() * getElementSize())
            throw new IllegalArgumentException(
                    "Illegal offset " + offset + " with index of " + index + " and length " + length());

        // TODO: fix this
        throw new UnsupportedOperationException("Deprecated set() call");
    }

    /**
     * Set an individual element
     *
     * @param index the index of the element
     * @param from  the element to get data from
     */
    @Deprecated
    protected void set(long index, long length, Pointer from) {
        set(index, length, from, 1);
    }

    @Override
    public void assign(Number value, long offset) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        //note here that the final put will take care of the offset
        super.assign(value,offset);



    }
    @Override
    public void assign(DataBuffer data) {
        //TODO: assign seems to have issue with the first value being bogus.
        //not clear why. We can see this when Nd4j.createBuffer (which calls getBuffer followed by setData)
        //TODO: it could have something to do with different setData implementations which are fairly new.
        //TODO: look for specific combinations of data that fail and fix one by one.
        allocator.memcpy(this, data);

    }

    @Override
    public void assign(long[] indices, float[] data, boolean contiguous, long inc) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length "
                    + length() + " where the indices are of length " + data.length);

        // TODO: eventually consider memcpy here
        for (int i = 0; i < indices.length; i++) {
            put(indices[i], data[i]);
        }


    }

    @Override
    public void assign(long[] indices, double[] data, boolean contiguous, long inc) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length "
                    + length() + " where the indices are of length " + data.length);

        // TODO: eventually consider memcpy here
        for (int i = 0; i < indices.length; i++) {
            put(indices[i], data[i]);
        }


    }


    /**
     * Set an individual element
     *
     * @param index the index of the element
     * @param from  the element to get data from
     */
    @Deprecated
    protected void set(long index, Pointer from) {
        set(index, 1, from);
    }

    @Override
    public void flush() {
        //
    }


    @Override
    public void destroy() {}

    @Override
    protected double getDoubleUnsynced(long index) {
        return super.getDouble(index);
    }

    @Override
    protected float getFloatUnsynced(long index) {
        return super.getFloat(index);
    }

    @Override
    protected long getLongUnsynced(long index) {
        return super.getLong(index);
    }

    @Override
    protected int getIntUnsynced(long index) {
        return super.getInt(index);
    }

    @Override
    public void write(DataOutputStream out) throws IOException {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        super.write(out);
    }

    @Override
    public void write(OutputStream dos) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        super.write(dos);
    }

    private void writeObject(java.io.ObjectOutputStream stream) throws IOException {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        stream.defaultWriteObject();
        write(stream);
    }

    private void readObject(java.io.ObjectInputStream stream) throws IOException, ClassNotFoundException {
        doReadObject(stream);
    }

    @Override
    public String toString() {
        if (ptrDataBuffer == null || ptrDataBuffer.isNull()) {
            return "CudaDataBuffer[DEALLOCATED, length=" + length + ", type=" + type + "]";
        }

        try {
            lazyAllocateHostPointer();
            AtomicAllocator.getInstance().synchronizeHostData(this);
            return super.toString();
        } catch (Exception e) {
            // If we still get an error (e.g., race condition with deallocation),
            // return a safe fallback string instead of propagating the exception
            return "CudaDataBuffer[ERROR: " + e.getMessage() + ", length=" + length + ", type=" + type + "]";
        }
    }

    @Override
    public boolean sameUnderlyingData(DataBuffer buffer) {
        return ptrDataBuffer.address() == ((BaseCudaDataBuffer) buffer).ptrDataBuffer.address();
    }


    @Override
    public void read(InputStream is, AllocationMode allocationMode, long length, DataType dataType) {
        if (allocationPoint == null) {
            initPointers(length, dataType, false);
        }
        super.read(is, allocationMode, length, dataType);
        this.allocationPoint.tickHostWrite();
    }


    /**
     * Initialize pointer and indexer based on the current data type
     *
     * @param currentType the current data type
     */
    @Override
    public void pointerIndexerByCurrentType(DataType currentType) {
        type = currentType;

        if (ptrDataBuffer == null) {
            ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(length(), type, true);
            this.deallocationId = ptrDataBuffer.getDeallocator() != null ?
                ptrDataBuffer.getDeallocator().getUniqueId() : -1;
        }

        actualizePointerAndIndexer();
    }

    //@Override
    public void read(DataInputStream s) {
        try {
            val savedMode = AllocationMode.valueOf(s.readUTF());
            allocationMode = AllocationMode.MIXED_DATA_TYPES;

            long locLength = 0;

            if (savedMode.ordinal() < 3)
                locLength = s.readInt();
            else
                locLength = s.readLong();

            boolean reallocate = locLength != length || indexer == null;
            length = locLength;

            val t = DataType.valueOf(s.readUTF());
            if (globalType == null && Nd4j.dataType() != null) {
                globalType = Nd4j.dataType();
            }

            if (t == DataType.COMPRESSED) {
                type = t;
                return;
            }

            this.elementSize = (byte) Nd4j.sizeOfDataType(t);
            this.allocationPoint = AtomicAllocator.getInstance().allocateMemory(this, new AllocationShape(length, elementSize, t), false);

            this.type = t;

            this.ptrDataBuffer = allocationPoint.getPtrDataBuffer();
            this.deallocationId = ptrDataBuffer != null && ptrDataBuffer.getDeallocator() != null ?
                ptrDataBuffer.getDeallocator().getUniqueId() : -1;

            switch (type) {
                case DOUBLE: {
                    this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asDoublePointer();
                    indexer = DoubleIndexer.create((DoublePointer) pointer);
                }
                break;
                case FLOAT: {
                    this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asFloatPointer();
                    indexer = FloatIndexer.create((FloatPointer) pointer);
                }
                break;
                case HALF: {
                    this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asShortPointer();
                    indexer = HalfIndexer.create((ShortPointer) pointer);
                }
                break;
                case LONG: {
                    this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asLongPointer();
                    indexer = LongIndexer.create((LongPointer) pointer);
                }
                break;
                case INT: {
                    this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asIntPointer();
                    indexer = IntIndexer.create((IntPointer) pointer);
                }
                break;
                case SHORT: {
                    this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asShortPointer();
                    indexer = ShortIndexer.create((ShortPointer) pointer);
                }
                break;
                case BFLOAT16: {
                    this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asShortPointer();
                    indexer = Bfloat16Indexer.create((ShortPointer) pointer);
                }
                break;
                case UBYTE: {
                    this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asBytePointer();
                    indexer = UByteIndexer.create((BytePointer) pointer);
                }
                break;
                case BYTE: {
                    this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asBytePointer();
                    indexer = ByteIndexer.create((BytePointer) pointer);
                }
                break;
                case BOOL: {
                    this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asBooleanPointer();
                    indexer = BooleanIndexer.create((BooleanPointer) pointer);
                }
                break;
                default:
                    throw new UnsupportedOperationException("Unsupported data type: " + type);
            }

            readContent(s, t, t);
            allocationPoint.tickHostWrite();

        } catch (Exception e) {
            throw new RuntimeException(e);
        }


        // we call sync to copy back data to host
        AtomicAllocator.getInstance().getFlowController().synchronizeToDevice(allocationPoint);
    }

    @Override
    public byte[] asBytes() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asBytes();
    }

    @Override
    public double[] asDouble() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asDouble();
    }

    @Override
    public float[] asFloat() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asFloat();
    }

    @Override
    public int[] asInt() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asInt();
    }

    @Override
    public long[] asLong() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asLong();
    }

    @Override
    public ByteBuffer asNio() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asNio();
    }

    @Override
    public DoubleBuffer asNioDouble() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asNioDouble();
    }

    @Override
    public FloatBuffer asNioFloat() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asNioFloat();
    }

    @Override
    public IntBuffer asNioInt() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asNioInt();
    }

    @Override
    public DataBuffer dup() {
        if (released.get()) {
            throw new IllegalStateException(
                "BUFFER_LIFECYCLE: dup() called on a CLOSED DataBuffer. " +
                "length=" + length + ", type=" + type +
                ". The buffer was freed but a caller is still trying to duplicate it. " +
                "Check for premature buffer deallocation in DSP output lifecycle.");
        }
        if (ptrDataBuffer == null) {
            throw new IllegalStateException(
                "BUFFER_LIFECYCLE: dup() called but ptrDataBuffer is null (buffer already released). " +
                "length=" + length + ", type=" + type);
        }
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        DataBuffer buffer = create(this.length);
        Pointer hostPtr = allocator.getHostPointer(this);
        if (hostPtr == null) {
            // Host pointer unavailable — fall back to device-side copy
            Pointer devPtr = allocator.getPointer(this);
            if (devPtr != null) {
                allocator.memcpyBlocking(buffer, new CudaPointer(devPtr.address()), this.length * elementSize, 0);
            }
        } else {
            allocator.memcpyBlocking(buffer, new CudaPointer(hostPtr.address()), this.length * elementSize, 0);
        }
        return buffer;
    }

    @Override
    public Number getNumber(long i) {
        synchronized (this) {
            ensureCorrectDevice();
            lazyAllocateHostPointer();
            allocator.synchronizeHostData(this);
            return super.getNumber(i);
        }
    }

    @Override
    public double getDouble(long i) {
        synchronized (this) {
            ensureCorrectDevice();
            lazyAllocateHostPointer();
            allocator.synchronizeHostData(this);
            return super.getDouble(i);
        }
    }

    @Override
    public long getLong(long i) {
        synchronized (this) {
            ensureCorrectDevice();
            lazyAllocateHostPointer();
            allocator.synchronizeHostData(this);
            return super.getLong(i);
        }
    }


    @Override
    public float getFloat(long i) {
        synchronized (this) {
            ensureCorrectDevice();
            lazyAllocateHostPointer();
            allocator.synchronizeHostData(this);
            return super.getFloat(i);
        }
    }

    @Override
    public int getInt(long ix) {
        synchronized (this) {
            ensureCorrectDevice();
            lazyAllocateHostPointer();
            allocator.synchronizeHostData(this);
            return super.getInt(ix);
        }
    }

    public void actualizePointerAndIndexer() {
        val cptr = ptrDataBuffer.primaryBuffer();

        // skip update if pointers are equal
        if (cptr != null && pointer != null && cptr.address() == pointer.address())
            return;

        val t = dataType();
        if (t == DataType.BOOL) {
            pointer = new PagedPointer(cptr, length).asBoolPointer();
            setIndexer(BooleanIndexer.create((BooleanPointer) pointer));
        } else if (t == DataType.UBYTE) {
            pointer = new PagedPointer(cptr, length).asBytePointer();
            setIndexer(UByteIndexer.create((BytePointer) pointer));
        } else if (t == DataType.BYTE) {
            pointer = new PagedPointer(cptr, length).asBytePointer();
            setIndexer(ByteIndexer.create((BytePointer) pointer));
        } else if (t == DataType.UINT16) {
            pointer = new PagedPointer(cptr, length).asShortPointer();
            setIndexer(UShortIndexer.create((ShortPointer) pointer));
        } else if (t == DataType.SHORT) {
            pointer = new PagedPointer(cptr, length).asShortPointer();
            setIndexer(ShortIndexer.create((ShortPointer) pointer));
        } else if (t == DataType.UINT32) {
            pointer = new PagedPointer(cptr, length).asIntPointer();
            setIndexer(UIntIndexer.create((IntPointer) pointer));
        } else if (t == DataType.INT) {
            pointer = new PagedPointer(cptr, length).asIntPointer();
            setIndexer(IntIndexer.create((IntPointer) pointer));
        } else if (t == DataType.UINT64) {
            pointer = new PagedPointer(cptr, length).asLongPointer();
            setIndexer(ULongIndexer.create((LongPointer) pointer));
        } else if (t == DataType.LONG) {
            pointer = new PagedPointer(cptr, length).asLongPointer();
            setIndexer(LongIndexer.create((LongPointer) pointer));
        } else if (t == DataType.BFLOAT16) {
            pointer = new PagedPointer(cptr, length).asShortPointer();
            setIndexer(Bfloat16Indexer.create((ShortPointer) pointer));
        } else if (t == DataType.HALF) {
            pointer = new PagedPointer(cptr, length).asShortPointer();
            setIndexer(HalfIndexer.create((ShortPointer) pointer));
        } else if (t == DataType.FLOAT) {
            pointer = new PagedPointer(cptr, length).asFloatPointer();
            setIndexer(FloatIndexer.create((FloatPointer) pointer));
        } else if (t == DataType.DOUBLE) {
            pointer = new PagedPointer(cptr, length).asDoublePointer();
            setIndexer(DoubleIndexer.create((DoublePointer) pointer));
        } else if (t == DataType.UTF8) {
            pointer = new PagedPointer(cptr, length()).asBytePointer();
            setIndexer(ByteIndexer.create((BytePointer) pointer));
        } else
            throw new IllegalArgumentException("Unknown datatype: " + dataType());
    }

    @Override
    public DataBuffer reallocate(long length) {
        val oldHostPointer = this.ptrDataBuffer.primaryBuffer();
        val oldDevicePointer = this.ptrDataBuffer.specialBuffer();

        if (isAttached()) {
            val capacity = length * getElementSize();

            if (oldDevicePointer != null && oldDevicePointer.address() != 0) {
                val nPtr = getParentWorkspace().alloc(capacity, MemoryKind.DEVICE, dataType(), false);
                NativeOpsHolder.getInstance().getDeviceNativeOps().memcpySync(nPtr, oldDevicePointer, length * getElementSize(), 3, null);
                this.ptrDataBuffer.setPrimaryBuffer(nPtr, length);

                allocationPoint.tickDeviceRead();
            }

            if (oldHostPointer != null && oldHostPointer.address() != 0) {
                val nPtr = getParentWorkspace().alloc(capacity, MemoryKind.HOST, dataType(), false);
                Pointer.memcpy(nPtr, oldHostPointer, this.length() * getElementSize());
                this.ptrDataBuffer.setPrimaryBuffer(nPtr, length);

                allocationPoint.tickHostRead();

                switch (dataType()) {
                    case BOOL:
                        pointer = nPtr.asBoolPointer();
                        indexer = BooleanIndexer.create((BooleanPointer) pointer);
                        break;
                    case UTF8:
                    case BYTE:
                    case UBYTE:
                        pointer = nPtr.asBytePointer();
                        indexer = ByteIndexer.create((BytePointer) pointer);
                        break;
                    case UINT16:
                    case SHORT:
                        pointer = nPtr.asShortPointer();
                        indexer = ShortIndexer.create((ShortPointer) pointer);
                        break;
                    case UINT32:
                        pointer = nPtr.asIntPointer();
                        indexer = UIntIndexer.create((IntPointer) pointer);
                        break;
                    case INT:
                        pointer = nPtr.asIntPointer();
                        indexer = IntIndexer.create((IntPointer) pointer);
                        break;
                    case DOUBLE:
                        pointer = nPtr.asDoublePointer();
                        indexer = DoubleIndexer.create((DoublePointer) pointer);
                        break;
                    case FLOAT:
                        pointer = nPtr.asFloatPointer();
                        indexer = FloatIndexer.create((FloatPointer) pointer);
                        break;
                    case HALF:
                        pointer = nPtr.asShortPointer();
                        indexer = HalfIndexer.create((ShortPointer) pointer);
                        break;
                    case BFLOAT16:
                        pointer = nPtr.asShortPointer();
                        indexer = Bfloat16Indexer.create((ShortPointer) pointer);
                        break;
                    case UINT64:
                    case LONG:
                        pointer = nPtr.asLongPointer();
                        indexer = LongIndexer.create((LongPointer) pointer);
                        break;
                }
            }


            workspaceGenerationId = getParentWorkspace().getGenerationId();
        } else {
            this.ptrDataBuffer.expand(length);
            val nPtr = new PagedPointer(this.ptrDataBuffer.primaryBuffer(), length);

            switch (dataType()) {
                case BOOL:
                    pointer = nPtr.asBoolPointer();
                    indexer = BooleanIndexer.create((BooleanPointer) pointer);
                    break;
                case UTF8:
                case BYTE:
                case UBYTE:
                    pointer = nPtr.asBytePointer();
                    indexer = ByteIndexer.create((BytePointer) pointer);
                    break;
                case UINT16:
                case SHORT:
                    pointer = nPtr.asShortPointer();
                    indexer = ShortIndexer.create((ShortPointer) pointer);
                    break;
                case UINT32:
                    pointer = nPtr.asIntPointer();
                    indexer = UIntIndexer.create((IntPointer) pointer);
                    break;
                case INT:
                    pointer = nPtr.asIntPointer();
                    indexer = IntIndexer.create((IntPointer) pointer);
                    break;
                case DOUBLE:
                    pointer = nPtr.asDoublePointer();
                    indexer = DoubleIndexer.create((DoublePointer) pointer);
                    break;
                case FLOAT:
                    pointer = nPtr.asFloatPointer();
                    indexer = FloatIndexer.create((FloatPointer) pointer);
                    break;
                case HALF:
                    pointer = nPtr.asShortPointer();
                    indexer = HalfIndexer.create((ShortPointer) pointer);
                    break;
                case BFLOAT16:
                    pointer = nPtr.asShortPointer();
                    indexer = Bfloat16Indexer.create((ShortPointer) pointer);
                    break;
                case UINT64:
                case LONG:
                    pointer = nPtr.asLongPointer();
                    indexer = LongIndexer.create((LongPointer) pointer);
                    break;
            }
        }

        this.underlyingLength = length;
        this.length = length;
        return this;
    }

    @Override
    public long capacity() {
        if (allocationPoint.getHostPointer() != null)
            return pointer.capacity();
        else
            return length;
    }

    @Override
    protected void release() {
        if (!released.get()) {
            released.set(true);
            ptrDataBuffer.closeBuffer();
            allocationPoint.setReleased(true);
        }
    }


    @Override
    public long getUniqueId() {
        return BASE_CUDA_DATA_BUFFER_OFFSET + allocationPoint.getObjectId();
    }

    /**
     * This method returns deallocator associated with this instance
     * @return
     */
    @Override
    public Deallocator deallocator() {
        if(deallocator != null)
            return deallocator;

        deallocator = new CudaDeallocator(this);
        return deallocator;

    }

    @Override
    public int targetDevice() {
        if (released.get()) {
            throw new IllegalStateException(
                "BUFFER_LIFECYCLE: targetDevice() called on a CLOSED DataBuffer. " +
                "The buffer was released via close()/release() but something still holds a reference. " +
                "length=" + length + ", type=" + type +
                ". Check for premature buffer deallocation — likely DSP output freed while still in use.");
        }
        if (ptrDataBuffer == null) {
            throw new IllegalStateException(
                "BUFFER_LIFECYCLE: targetDevice() called but ptrDataBuffer is null. " +
                "The buffer was released via close()/release() (ptrDataBuffer set to null). " +
                "length=" + length + ", type=" + type);
        }
        return AtomicAllocator.getInstance().getAllocationPoint(this).getDeviceId();
    }

    @Override
    public void syncToPrimary(){
        if (released.get() || ptrDataBuffer == null) {
            throw new IllegalStateException(
                "BUFFER_LIFECYCLE: syncToPrimary() called on a " +
                (released.get() ? "CLOSED" : "null-ptrDataBuffer") + " DataBuffer. " +
                "length=" + length + ", type=" + type);
        }
        ptrDataBuffer.syncToPrimary();
    }

    @Override
    public void syncToSpecial(){
        if (released.get() || ptrDataBuffer == null) {
            throw new IllegalStateException(
                "BUFFER_LIFECYCLE: syncToSpecial() called on a " +
                (released.get() ? "CLOSED" : "null-ptrDataBuffer") + " DataBuffer. " +
                "length=" + length + ", type=" + type);
        }
        ptrDataBuffer.syncToSpecial();
    }

    // ========================
    // HybridDataBuffer Implementation
    // ========================

    @Override
    public boolean isHybrid() {
        return true;
    }

    @Override
    public HybridDataBuffer asHybrid() {
        return this;
    }

    @Override
    public DeviceDescriptor getOwnerDevice() {
        if (ownerDevice == null) {
            if (released.get()) {
                throw new IllegalStateException(
                    "BUFFER_LIFECYCLE: getOwnerDevice() called on a CLOSED DataBuffer. " +
                    "length=" + length + ", type=" + type +
                    ". The buffer was freed but a caller is still trying to determine its device. " +
                    "This typically means a DSP output was freed by closeSlotArrayCache/releaseGpuIntermediates " +
                    "before TOKEN_SAMPLE or dup() could use it.");
            }
            int deviceId = targetDevice();
            ownerDevice = DeviceDescriptor.cuda(deviceId);
        }
        return ownerDevice;
    }

    @Override
    public void setOwnerDevice(DeviceDescriptor device) {
        validatePinnedDevice(device);
        this.ownerDevice = device;
    }

    @Override
    public DeviceDescriptor getPinnedDevice() {
        return pinnedDevice;
    }

    @Override
    public void pinTo(DeviceDescriptor device) {
        if (device == null) {
            unpin();
            return;
        }

        DeviceDescriptor previousPinned = pinnedDevice;
        if (previousPinned != null && !previousPinned.getDeviceId().equals(device.getDeviceId())) {
            pinnedDevice = null;
        }

        // Ensure data is available on the target device before pinning
        ensureAvailableOn(device);
        pinnedDevice = device;
        ownerDevice = device;

        // For CPU pinning, mark CPU as valid and potentially invalidate GPU copy
        if (device.getDeviceType() == DeviceType.CPU) {
            cpuValid = true;
            // When pinned to CPU, GPU copy may become stale if modified
        }
    }

    @Override
    public void unpin() {
        pinnedDevice = null;
    }

    @Override
    public boolean isValidOn(DeviceDescriptor device) {
        if (device == null) {
            return false;
        }
        // Check explicit validity flags
        if (device.getDeviceType() == DeviceType.CPU) {
            return cpuValid;
        } else if (device.getDeviceType().isGpu()) {
            return gpuValid;
        }
        return false;
    }

    @Override
    public void markValidOn(DeviceDescriptor device) {
        if (device == null) {
            return;
        }
        if (device.getDeviceType() == DeviceType.CPU) {
            cpuValid = true;
            hostDirty = false;
        } else if (device.getDeviceType().isGpu()) {
            validatePinnedDevice(device);
            gpuValid = true;
            deviceDirty = false;
            ownerDevice = device;
        }
    }

    @Override
    public void markInvalidOn(DeviceDescriptor device) {
        if (device == null) {
            return;
        }
        if (device.getDeviceType() == DeviceType.CPU) {
            cpuValid = false;
            hostDirty = true;
        } else if (device.getDeviceType().isGpu()) {
            gpuValid = false;
            deviceDirty = true;
        }
    }

    @Override
    public void ensureAvailableOn(DeviceDescriptor device) {
        if (device == null || ptrDataBuffer == null) {
            return;
        }

        validatePinnedDevice(device);

        if (device.getDeviceType() == DeviceType.CPU) {
            // Need data on CPU/host
            if (!cpuValid) {
                // Sync from GPU to CPU
                syncToPrimary();
                cpuValid = true;
                deviceDirty = false;
            }
        } else if (device.getDeviceType().isGpu()) {
            // Need data on GPU
            int targetDeviceIndex = device.getDeviceIndex();
            int currentNativeDevice = targetDevice();

            if (currentNativeDevice != -1 && currentNativeDevice != targetDeviceIndex) {
                // Check if we need to migrate or if P2P is enough
                if (!Nd4j.getAffinityManager().isCrossDeviceAccessSupported()) {
                    // Save current device context — migrate requires switching to target
                    // device, but we must restore the caller's context afterward.
                    // Without this, the calling thread's CUDA device is permanently changed
                    // to the target device, causing subsequent ops to execute on the wrong
                    // device (error 700: illegal memory access).
                    int callerDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
                    DeviceMemoryManager.getInstance().switchDevice(targetDeviceIndex, "BaseCudaDataBuffer.ensureAvailableOn", "migrate-to-target");
                    // Call native migrate to move buffer to target device
                    ptrDataBuffer.migrate();
                    gpuValid = true;
                    hostDirty = false;
                    // Re-read actual device from native layer — migrate's internal
                    // failover may have routed to a different device than requested
                    int actualDevice = targetDevice();
                    ownerDevice = DeviceDescriptor.cuda(actualDevice);
                    // Restore caller's device context so the thread doesn't stay on
                    // the target device after the transfer
                    if (callerDevice != targetDeviceIndex) {
                        DeviceMemoryManager.getInstance().switchDevice(callerDevice, "BaseCudaDataBuffer.ensureAvailableOn", "restore-caller-context");
                    }
                    return;
                } else {
                    // P2P is available - treat device data as valid on target
                    gpuValid = true;
                    hostDirty = false;
                }
            } else if (!gpuValid) {
                // Sync from CPU to GPU
                syncToSpecial();
                gpuValid = true;
                hostDirty = false;
            }
            // Update owner to target device
            ownerDevice = device;
        }
    }

    private void validatePinnedDevice(DeviceDescriptor device) {
        DeviceDescriptor pinned = pinnedDevice;
        if (pinned == null || device == null) {
            return;
        }

        // If pinned to a specific device, ensure the requested device matches
        if (!pinned.getDeviceId().equals(device.getDeviceId())) {
            // Special case: if pinned to CPU, allow read-only access from GPU
            // (the data will be synced from CPU to GPU as needed)
            if (pinned.getDeviceType() == DeviceType.CPU && device.getDeviceType().isGpu()) {
                // Allow GPU to read from CPU-pinned buffer - will sync as needed
                return;
            }
            // Special case: if pinned to GPU, allow read-only access from CPU
            // (the data will be synced from GPU to CPU as needed)
            if (pinned.getDeviceType().isGpu() && device.getDeviceType() == DeviceType.CPU) {
                // Allow CPU to read from GPU-pinned buffer - will sync as needed
                return;
            }
            // GPU-to-GPU mismatch when pinned
            throw new IllegalStateException(
                    "Buffer is pinned to " + pinned.getDeviceId()
                            + " but requested device is " + device.getDeviceId()
                            + ". Unpin or repin before transferring.");
        }
    }

    public long getDeviceAddress(DeviceDescriptor device) {
        if (device == null || ptrDataBuffer == null) {
            return 0;
        }

        validatePinnedDevice(device);

        // Handle CPU device request - return host/primary buffer address
        if (device.getDeviceType() == DeviceType.CPU) {
            ensureAvailableOn(device);
            Pointer primary = ptrDataBuffer.primaryBuffer();
            return primary != null ? primary.address() : 0;
        }

        // Handle GPU device request
        int currentDeviceIndex = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int requestedDeviceIndex = device.getDeviceIndex();
        int bufferDeviceIndex = this.getNativeDeviceId();

        if (bufferDeviceIndex != -1 && bufferDeviceIndex != requestedDeviceIndex) {
            // Check if P2P access is available for direct access
            if (Nd4j.getAffinityManager().isCrossDeviceAccessSupported()) {
                // P2P available - can use current buffer address directly
                // The CUDA driver handles peer access transparently
                Pointer special = ptrDataBuffer.specialBuffer();
                return special != null ? special.address() : 0;
            } else {
                // No P2P - need to migrate data to the target device
                // Ensure data is synced to host first, then allocate on target device
                ensureAvailableOn(DeviceDescriptor.cpu());

                // Update owner device and migrate
                ownerDevice = device;

                // The next syncToSpecial call will allocate on the new device
                gpuValid = false;
                ensureAvailableOn(device);
            }
        } else if (!gpuValid) {
            // Ensure GPU data is valid
            ensureAvailableOn(device);
        }

        // Return device/special buffer address
        Pointer special = ptrDataBuffer.specialBuffer();
        return special != null ? special.address() : 0;
    }


    @Override
    public MultiBackendWorkspace getMultiBackendWorkspace() {
        return multiBackendWorkspace;
    }

    @Override
    public void attachToWorkspace(MultiBackendWorkspace workspace) {
        this.multiBackendWorkspace = workspace;
    }

    @Override
    public void detachFromWorkspace() {
        this.multiBackendWorkspace = null;
    }

    @Override
    public void allocateOnDevice(DeviceDescriptor device, long requiredSize) {
        if (device == null || !device.getDeviceType().isGpu()) {
            return;
        }

        int targetDeviceId = device.getDeviceIndex();
        int currentDeviceId = targetDevice();
        if (targetDeviceId == currentDeviceId) {
            return; // already on target device
        }

        // Switch to target device and migrate the native buffer
        DeviceMemoryManager.getInstance().switchDevice(targetDeviceId, "BaseCudaDataBuffer", "allocateOnDevice");
        ptrDataBuffer.migrate();

        // Update tracking
        this.ownerDevice = device;
        this.gpuValid = true;
        this.hostDirty = false;
    }

    @Override
    public boolean isHostDirty() {
        return hostDirty;
    }

    @Override
    public boolean isDeviceDirty() {
        return deviceDirty;
    }

    @Override
    public void markHostDirty() {
        this.hostDirty = true;
    }

    @Override
    public void markDeviceDirty() {
        this.deviceDirty = true;
    }
}
