/*
 * ******************************************************************************
 * *
 * * This program and the accompanying materials are made available under the
 * * terms of the Apache License, Version 2.0 which is available at
 * * https://www.apache.org/licenses/LICENSE-2.0.
 * *
 * * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */
package org.nd4j.linalg.vulkan;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.HybridDataBuffer;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.MultiBackendWorkspace;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.vulkan.bindings.Nd4jVulkan;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Collection;

/**
 * Vulkan-owned dual-plane data buffer.
 *
 * <p>This is the Vulkan counterpart of CUDA's device buffer boundary. The Java
 * object owns the framework-facing datatype and host indexer while
 * {@link OpaqueDataBuffer} owns the native primary/special state machine. Host
 * memory is staging memory; it is not permission to execute an unsupported op
 * on the CPU.</p>
 */
public class VulkanDataBuffer extends BaseDataBuffer implements Deallocatable, HybridDataBuffer {
    private final VulkanRuntime runtime = VulkanRuntime.getInstance();
    private final Nd4jVulkan nativeOps = runtime.nativeOps();
    private final VulkanAffinityManager affinityManager = runtime.affinityManager();
    private final long instanceId = runtime.deallocatorService().nextValue();

    private volatile DeviceDescriptor ownerDevice;
    private volatile DeviceDescriptor pinnedDevice;
    private long numWords = -1;

    public VulkanDataBuffer() {
        this.type = DataType.UNKNOWN;
        this.allocationMode = AllocationMode.MIXED_DATA_TYPES;
    }

    public VulkanDataBuffer(DataType dataType, long length, boolean initialize) {
        init(dataType, length, initialize, null, false);
    }

    public VulkanDataBuffer(DataType dataType, long length, boolean initialize, MemoryWorkspace workspace) {
        init(dataType, length, initialize, workspace, false);
    }

    private VulkanDataBuffer(DataType dataType, long physicalLength, boolean initialize,
                             MemoryWorkspace workspace, boolean rawStringStorage) {
        init(dataType, physicalLength, initialize, workspace, rawStringStorage);
    }

    public VulkanDataBuffer(DataType dataType, Pointer primary, Indexer suppliedIndexer, long length) {
        this(dataType, primary, null, suppliedIndexer, length);
    }

    public VulkanDataBuffer(DataType dataType, Pointer primary, Pointer special,
                            Indexer suppliedIndexer, long length) {
        requireSupportedType(dataType);
        if (length < 0) {
            throw new IllegalArgumentException("Length must be >= 0");
        }
        this.type = dataType;
        initTypeAndSize();
        this.length = length;
        this.underlyingLength = length;
        this.allocationMode = AllocationMode.MIXED_DATA_TYPES;
        this.ptrDataBuffer = OpaqueDataBuffer.externalizedDataBuffer(
                length, dataType, primary, special, VulkanRuntime.getInstance());
        updateDeallocator();
        if (suppliedIndexer != null) {
            this.pointer = primary;
            this.indexer = suppliedIndexer;
        } else {
            bindPrimaryPointer();
        }
        nativeOps.dbTickHostWrite(ptrDataBuffer);
        this.ownerDevice = special == null
                ? DeviceDescriptor.cpu()
                : vulkanDevice(targetDevice());
    }

    public VulkanDataBuffer(ByteBuffer source, DataType dataType, long length) {
        this(dataType, length, false, null, isStringType(dataType));
        long bytes = Math.multiplyExact(length, getElementSize());
        if (bytes > 0) {
            ByteBuffer view = source.slice();
            if (view.remaining() < bytes) {
                throw new IllegalArgumentException("ByteBuffer has " + view.remaining()
                        + " bytes but " + bytes + " are required");
            }
            Pointer.memcpy(ptrDataBuffer.primaryBuffer(), new BytePointer(view), bytes);
            hostWritten();
        }
    }

    static VulkanDataBuffer fromStrings(Collection<String> strings, DataType dataType) {
        requireStringType(dataType);
        byte[][] encoded = new byte[strings.size()][];
        long payloadBytes = 0;
        int i = 0;
        for (String value : strings) {
            byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
            encoded[i++] = bytes;
            payloadBytes = Math.addExact(payloadBytes, bytes.length);
        }

        long headerBytes = Math.multiplyExact(strings.size() + 1L, Long.BYTES);
        VulkanDataBuffer buffer = new VulkanDataBuffer(
                dataType, Math.addExact(headerBytes, payloadBytes), false, null, true);
        buffer.numWords = strings.size();

        LongPointer offsets = new LongPointer(buffer.ptrDataBuffer.primaryBuffer());
        BytePointer bytes = new BytePointer(buffer.ptrDataBuffer.primaryBuffer());
        long offset = 0;
        for (i = 0; i < encoded.length; i++) {
            offsets.put(i, offset);
            byte[] word = encoded[i];
            if (word.length > 0) {
                bytes.position(headerBytes + offset).put(word);
            }
            offset += word.length;
        }
        offsets.put(encoded.length, offset);
        bytes.position(0);
        buffer.hostWritten();
        return buffer;
    }

    static VulkanDataBuffer fromEncodedStrings(byte[] data, long numWords) {
        if (numWords < 0) {
            throw new IllegalArgumentException("Number of strings must be >= 0");
        }
        VulkanDataBuffer buffer = new VulkanDataBuffer(
                DataType.UTF8, data.length, false, null, true);
        buffer.numWords = numWords;
        if (data.length > 0) {
            new BytePointer(buffer.ptrDataBuffer.primaryBuffer()).put(data);
            buffer.hostWritten();
        }
        return buffer;
    }

    private void init(DataType dataType, long requestedLength, boolean initialize,
                      MemoryWorkspace workspace, boolean rawStringStorage) {
        requireSupportedType(dataType);
        if (requestedLength < 0) {
            throw new IllegalArgumentException("Length must be >= 0");
        }

        this.type = dataType;
        initTypeAndSize();
        this.allocationMode = AllocationMode.MIXED_DATA_TYPES;

        long physicalLength = requestedLength;
        if (isStringType(dataType) && !rawStringStorage) {
            this.numWords = requestedLength;
            physicalLength = Math.multiplyExact(requestedLength + 1L, Long.BYTES);
        }
        this.length = physicalLength;
        this.underlyingLength = physicalLength;

        if (workspace == null) {
            ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(
                    physicalLength, dataType, true, VulkanRuntime.getInstance());
            updateDeallocator();
        } else {
            long bytes = Math.multiplyExact(physicalLength, getElementSize());
            Pointer primary = workspace.alloc(bytes, MemoryKind.HOST, dataType, initialize);
            Pointer special = workspace.alloc(bytes, MemoryKind.DEVICE, dataType, false);
            ptrDataBuffer = OpaqueDataBuffer.workspaceDataBuffer(
                    physicalLength, dataType, primary, special, VulkanRuntime.getInstance());
            attached = true;
            parentWorkspace = workspace;
            workspaceGenerationId = workspace.getGenerationId();
        }

        bindPrimaryPointer();
        if (initialize && physicalLength > 0) {
            Pointer.memset(pointer, 0, Math.multiplyExact(physicalLength, getElementSize()));
            hostWritten();
            ptrDataBuffer.syncToSpecial();
        }
        ownerDevice = vulkanDevice(targetDevice());
    }

    private static void requireSupportedType(DataType dataType) {
        if (dataType == null || dataType == DataType.UNKNOWN) {
            throw new IllegalArgumentException("A concrete DataType is required for a Vulkan buffer");
        }
        if (dataType == DataType.COMPRESSED) {
            throw new UnsupportedOperationException(
                    "Compressed buffers require a codec-owned storage contract");
        }
    }

    private static void requireStringType(DataType dataType) {
        if (!isStringType(dataType)) {
            throw new IllegalArgumentException("Expected UTF8, UTF16, or UTF32, got " + dataType);
        }
    }

    private static boolean isStringType(DataType dataType) {
        return dataType == DataType.UTF8
                || dataType == DataType.UTF16
                || dataType == DataType.UTF32;
    }

    boolean hasVariableLengthStringStorage() {
        return isStringType(type);
    }

    long stringElementCount() {
        if (!hasVariableLengthStringStorage()) {
            throw new IllegalStateException("Buffer is not string storage");
        }
        return numWords;
    }

    private static DeviceDescriptor vulkanDevice(int index) {
        if (index < 0) {
            throw new IllegalArgumentException("Vulkan device index must be non-negative: " + index);
        }
        return DeviceDescriptor.accelerator("vulkan", DeviceType.VULKAN_GPU, index);
    }

    private void checkNativeError(String action) {
        int errorCode = nativeOps.lastErrorCode();
        if (errorCode != 0) {
            String message = nativeOps.lastErrorMessage();
            nativeOps.clearLastError();
            throw new IllegalStateException(
                    "Vulkan native error while " + action + " (" + errorCode + "): " + message);
        }
    }

    private void updateDeallocator() {
        this.deallocationId = ptrDataBuffer != null && ptrDataBuffer.getDeallocator() != null
                ? ptrDataBuffer.getDeallocator().getUniqueId()
                : instanceId;
    }

    @Override
    protected void initTypeAndSize() {
        if (type == null || type == DataType.UNKNOWN) {
            return;
        }
        int width = type.width();
        this.elementSize = (byte) (isStringType(type) ? 1 : (width > 0 ? width : 1));
    }

    private void bindPrimaryPointer() {
        if (length == 0 || ptrDataBuffer == null) {
            this.pointer = null;
            this.indexer = null;
            return;
        }
        Pointer primary = ptrDataBuffer.primaryBuffer();
        if (primary == null || primary.isNull()) {
            nativeOps.dbAllocatePrimaryBuffer(ptrDataBuffer);
            primary = ptrDataBuffer.primaryBuffer();
        }
        PagedPointer typed = new PagedPointer(primary, length);
        switch (type) {
            case DOUBLE:
                pointer = typed.asDoublePointer();
                indexer = DoubleIndexer.create((DoublePointer) pointer);
                break;
            case FLOAT:
                pointer = typed.asFloatPointer();
                indexer = FloatIndexer.create((FloatPointer) pointer);
                break;
            case HALF:
                pointer = typed.asShortPointer();
                indexer = HalfIndexer.create((ShortPointer) pointer);
                break;
            case BFLOAT16:
                pointer = typed.asShortPointer();
                indexer = Bfloat16Indexer.create((ShortPointer) pointer);
                break;
            case LONG:
                pointer = typed.asLongPointer();
                indexer = LongIndexer.create((LongPointer) pointer);
                break;
            case UINT64:
                pointer = typed.asLongPointer();
                indexer = ULongIndexer.create((LongPointer) pointer);
                break;
            case INT:
                pointer = typed.asIntPointer();
                indexer = IntIndexer.create((IntPointer) pointer);
                break;
            case UINT32:
                pointer = typed.asIntPointer();
                indexer = UIntIndexer.create((IntPointer) pointer);
                break;
            case SHORT:
                pointer = typed.asShortPointer();
                indexer = ShortIndexer.create((ShortPointer) pointer);
                break;
            case UINT16:
                pointer = typed.asShortPointer();
                indexer = UShortIndexer.create((ShortPointer) pointer);
                break;
            case UBYTE:
                pointer = typed.asBytePointer();
                indexer = UByteIndexer.create((BytePointer) pointer);
                break;
            case BOOL:
                pointer = typed.asBoolPointer();
                indexer = BooleanIndexer.create((BooleanPointer) pointer);
                break;
            case BYTE:
            case FLOAT8:
            case FLOAT8_E5M2:
            case UTF8:
            case UTF16:
            case UTF32:
                pointer = typed.asBytePointer();
                indexer = ByteIndexer.create((BytePointer) pointer);
                break;
            default:
                throw new UnsupportedOperationException(
                        "No Vulkan host staging indexer for " + type);
        }
    }

    private void synchronizeHostView() {
        if (ptrDataBuffer == null || length == 0) {
            return;
        }
        ptrDataBuffer.syncToPrimary();
        Pointer primary = ptrDataBuffer.primaryBuffer();
        if (pointer == null || pointer.address() != primary.address()) {
            bindPrimaryPointer();
        }
    }

    private void hostWritten() {
        nativeOps.dbTickHostWrite(ptrDataBuffer);
        ownerDevice = DeviceDescriptor.cpu();
    }

    @Override
    public Pointer pointer() {
        synchronizeHostView();
        return super.pointer();
    }

    @Override
    public Indexer indexer() {
        synchronizeHostView();
        return super.indexer();
    }

    @Override
    public Pointer addressPointer() {
        return pointer();
    }

    @Override
    public long platformAddress() {
        syncToSpecial();
        Pointer special = ptrDataBuffer.specialBuffer();
        return special == null ? 0 : special.address();
    }

    @Override
    public void pointerIndexerByCurrentType(DataType currentType) {
        requireSupportedType(currentType);
        this.type = currentType;
        initTypeAndSize();
        if (ptrDataBuffer == null) {
            ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(
                    length, currentType, true, VulkanRuntime.getInstance());
            updateDeallocator();
        }
        bindPrimaryPointer();
    }

    @Override
    protected DataBuffer create(long length) {
        return new VulkanDataBuffer(type, length, false);
    }

    @Override
    public DataBuffer create(double[] data) {
        VulkanDataBuffer result = new VulkanDataBuffer(type, data.length, false);
        result.put(data);
        return result;
    }

    @Override
    public DataBuffer create(float[] data) {
        VulkanDataBuffer result = new VulkanDataBuffer(type, data.length, false);
        result.put(data);
        return result;
    }

    @Override
    public DataBuffer create(int[] data) {
        VulkanDataBuffer result = new VulkanDataBuffer(type, data.length, false);
        result.put(data);
        return result;
    }

    @Override
    protected double getDoubleUnsynced(long index) {
        synchronizeHostView();
        return super.getDouble(index);
    }

    @Override
    protected float getFloatUnsynced(long index) {
        synchronizeHostView();
        return super.getFloat(index);
    }

    @Override
    protected long getLongUnsynced(long index) {
        synchronizeHostView();
        return super.getLong(index);
    }

    @Override
    protected int getIntUnsynced(long index) {
        synchronizeHostView();
        return super.getInt(index);
    }

    @Override
    public double getDouble(long index) {
        synchronizeHostView();
        return super.getDouble(index);
    }

    @Override
    public float getFloat(long index) {
        synchronizeHostView();
        return super.getFloat(index);
    }

    @Override
    public long getLong(long index) {
        synchronizeHostView();
        return super.getLong(index);
    }

    @Override
    public int getInt(long index) {
        synchronizeHostView();
        return super.getInt(index);
    }

    @Override
    public Number getNumber(long index) {
        synchronizeHostView();
        return super.getNumber(index);
    }

    private void beforeHostWrite() {
        synchronizeHostView();
    }

    @Override
    public void put(long index, float value) {
        beforeHostWrite();
        super.put(index, value);
        hostWritten();
    }

    @Override
    public void put(long index, double value) {
        beforeHostWrite();
        super.put(index, value);
        hostWritten();
    }

    @Override
    public void put(long index, short value) {
        beforeHostWrite();
        super.put(index, value);
        hostWritten();
    }

    @Override
    public void put(long index, int value) {
        beforeHostWrite();
        super.put(index, value);
        hostWritten();
    }

    @Override
    public void put(long index, boolean value) {
        beforeHostWrite();
        super.put(index, value);
        hostWritten();
    }

    @Override
    public void put(long index, long value) {
        beforeHostWrite();
        super.put(index, value);
        hostWritten();
    }

    @Override
    public void put(float[] values) {
        beforeHostWrite();
        super.put(values);
        hostWritten();
    }

    @Override
    public void put(double[] values) {
        beforeHostWrite();
        super.put(values);
        hostWritten();
    }

    @Override
    public void put(int[] values) {
        beforeHostWrite();
        super.put(values);
        hostWritten();
    }

    @Override
    public void put(boolean[] values) {
        beforeHostWrite();
        super.put(values);
        hostWritten();
    }

    @Override
    public void put(short[] values) {
        beforeHostWrite();
        super.put(values);
        hostWritten();
    }

    @Override
    public void put(byte[] values) {
        beforeHostWrite();
        super.put(values);
        hostWritten();
    }

    @Override
    public void put(long[] values) {
        beforeHostWrite();
        super.put(values);
        hostWritten();
    }

    public String getString(long index) {
        requireStringType(type);
        synchronizeHostView();
        if (numWords < 0) {
            throw new IllegalStateException("String count is unavailable for this external buffer");
        }
        if (index < 0 || index >= numWords) {
            throw new IndexOutOfBoundsException(
                    "String index " + index + " outside [0," + numWords + ")");
        }

        LongPointer offsets = new LongPointer(ptrDataBuffer.primaryBuffer());
        long start = offsets.get(index);
        long end = offsets.get(index + 1);
        if (end < start || end - start > Integer.MAX_VALUE) {
            throw new IllegalStateException("Invalid encoded string offsets: " + start + ".." + end);
        }

        byte[] bytes = new byte[(int) (end - start)];
        long headerBytes = Math.multiplyExact(numWords + 1L, Long.BYTES);
        new BytePointer(ptrDataBuffer.primaryBuffer()).position(headerBytes + start).get(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    @Override
    public DataBuffer dup() {
        return duplicateToDevice(targetDevice());
    }

    VulkanDataBuffer duplicateToDevice(int destinationDevice) {
        int sourceDevice = targetDevice();
        if (!affinityManager.isPeerCopyAllowed(sourceDevice, destinationDevice)) {
            throw new UnsupportedOperationException(
                    "Vulkan peer copy is unavailable from device " + sourceDevice
                            + " to device " + destinationDevice);
        }

        ensureAvailableOn(vulkanDevice(sourceDevice));
        checkNativeError("preparing source buffer for device copy");

        int previousDevice = affinityManager.getDeviceForCurrentThread();
        VulkanDataBuffer result;
        try {
            if (previousDevice != destinationDevice) {
                affinityManager.setDeviceForCurrentThread(destinationDevice);
            }
            result = isStringType(type)
                    ? new VulkanDataBuffer(type, length, false, null, true)
                    : new VulkanDataBuffer(type, length, false);
            result.numWords = numWords;
        } finally {
            if (previousDevice != destinationDevice) {
                affinityManager.setDeviceForCurrentThread(previousDevice);
            }
        }

        nativeOps.dbAsyncCrossDeviceCopy(result.ptrDataBuffer, ptrDataBuffer, null);
        checkNativeError("copying Vulkan buffer from device " + sourceDevice
                + " to device " + destinationDevice);
        result.ownerDevice = vulkanDevice(destinationDevice);
        return result;
    }

    @Override
    public DataBuffer reallocate(long newLength) {
        if (newLength < 0) {
            throw new IllegalArgumentException("Length must be >= 0");
        }
        if (attached) {
            throw new UnsupportedOperationException(
                    "Workspace-owned Vulkan buffers cannot be reallocated in place");
        }
        ptrDataBuffer.expand(newLength);
        this.length = newLength;
        this.underlyingLength = newLength;
        bindPrimaryPointer();
        return this;
    }

    @Override
    public void syncToPrimary() {
        synchronizeHostView();
    }

    @Override
    public void syncToSpecial() {
        if (ptrDataBuffer != null && length > 0) {
            ptrDataBuffer.syncToSpecial();
            ownerDevice = vulkanDevice(targetDevice());
        }
    }

    @Override
    public long getUniqueId() {
        return deallocationId >= 0 ? deallocationId : instanceId;
    }

    @Override
    public Deallocator deallocator() {
        return deallocator;
    }

    @Override
    public int targetDevice() {
        if (ptrDataBuffer == null) {
            return affinityManager.getDeviceForCurrentThread();
        }
        int device = ptrDataBuffer.deviceId();
        return device >= 0 ? device : affinityManager.getDeviceForCurrentThread();
    }

    @Override
    public boolean shouldDeAllocate() {
        return !released.get() && !attached && !isConstant();
    }

    @Override
    public DeviceDescriptor getOwnerDevice() {
        if (ownerDevice == null) {
            ownerDevice = vulkanDevice(targetDevice());
        }
        return ownerDevice;
    }

    @Override
    public void setOwnerDevice(DeviceDescriptor device) {
        validatePlacement(device);
        this.ownerDevice = device;
    }

    @Override
    public DeviceDescriptor getPinnedDevice() {
        return pinnedDevice;
    }

    @Override
    public void pinTo(DeviceDescriptor device) {
        validatePlacement(device);
        ensureAvailableOn(device);
        pinnedDevice = device;
    }

    @Override
    public void unpin() {
        pinnedDevice = null;
    }

    @Override
    public boolean isValidOn(DeviceDescriptor device) {
        validatePlacement(device);
        if (device.getDeviceType() == DeviceType.CPU) {
            return nativeOps.dbIsPrimaryActual(ptrDataBuffer);
        }
        return targetDevice() == device.getDeviceIndex()
                && nativeOps.dbIsSpecialActual(ptrDataBuffer);
    }

    @Override
    public void markValidOn(DeviceDescriptor device) {
        validatePlacement(device);
        if (device.getDeviceType() == DeviceType.CPU) {
            nativeOps.dbTickHostWrite(ptrDataBuffer);
        } else {
            nativeOps.dbTickDeviceWrite(ptrDataBuffer);
        }
        ownerDevice = device;
    }

    @Override
    public void markInvalidOn(DeviceDescriptor device) {
        validatePlacement(device);
        if (device.getDeviceType() == DeviceType.CPU) {
            nativeOps.dbTickDeviceWrite(ptrDataBuffer);
            ownerDevice = vulkanDevice(targetDevice());
        } else {
            nativeOps.dbTickHostWrite(ptrDataBuffer);
            ownerDevice = DeviceDescriptor.cpu();
        }
    }

    @Override
    public void ensureAvailableOn(DeviceDescriptor device) {
        validatePlacement(device);
        DeviceDescriptor pin = pinnedDevice;
        if (pin != null && !pin.getDeviceId().equals(device.getDeviceId())) {
            throw new IllegalStateException(
                    "Buffer is pinned to " + pin.getDeviceId() + ", not " + device.getDeviceId());
        }

        if (device.getDeviceType() == DeviceType.CPU) {
            syncToPrimary();
            return;
        }

        int previous = affinityManager.getDeviceForCurrentThread();
        try {
            if (previous != device.getDeviceIndex()) {
                nativeOps.setDevice(device.getDeviceIndex());
            }
            if (targetDevice() != device.getDeviceIndex()) {
                ptrDataBuffer.migrate();
            }
            ptrDataBuffer.syncToSpecial();
            ownerDevice = device;
        } finally {
            if (previous != device.getDeviceIndex()) {
                nativeOps.setDevice(previous);
            }
        }
    }

    private static void validatePlacement(DeviceDescriptor device) {
        if (device == null) {
            throw new IllegalArgumentException("Device must not be null");
        }
        if (device.getDeviceType() != DeviceType.CPU
                && device.getDeviceType() != DeviceType.VULKAN_GPU) {
            throw new IllegalArgumentException(
                    "Vulkan buffers can reside only in host staging memory or on Vulkan devices: "
                            + device.getDeviceId());
        }
    }

    @Override
    public long getDeviceAddress(DeviceDescriptor device) {
        ensureAvailableOn(device);
        Pointer result = device.getDeviceType() == DeviceType.CPU
                ? ptrDataBuffer.primaryBuffer()
                : ptrDataBuffer.specialBuffer();
        return result == null ? 0 : result.address();
    }

    @Override
    public MultiBackendWorkspace getMultiBackendWorkspace() {
        return null;
    }

    @Override
    public void attachToWorkspace(MultiBackendWorkspace workspace) {
        if (workspace == null) {
            throw new IllegalArgumentException("Workspace must not be null");
        }
        throw new UnsupportedOperationException(
                "Attaching an existing Vulkan buffer requires workspace-owned host and device allocations; "
                        + "metadata-only attach is unsafe");
    }

    @Override
    public void detachFromWorkspace() {
        throw new UnsupportedOperationException(
                "Detaching a Vulkan buffer requires an independent device allocation and copy; "
                        + "metadata-only detach is unsafe");
    }

    @Override
    public void allocateOnDevice(DeviceDescriptor device, long requiredSize) {
        if (requiredSize > Math.multiplyExact(length, getElementSize())) {
            throw new IllegalArgumentException(
                    "Requested allocation exceeds this buffer's logical size");
        }
        ensureAvailableOn(device);
    }

    @Override
    public boolean isHostDirty() {
        return nativeOps.dbIsPrimaryActual(ptrDataBuffer)
                && !nativeOps.dbIsSpecialActual(ptrDataBuffer);
    }

    @Override
    public boolean isDeviceDirty() {
        return nativeOps.dbIsSpecialActual(ptrDataBuffer)
                && !nativeOps.dbIsPrimaryActual(ptrDataBuffer);
    }

    @Override
    public void markHostDirty() {
        hostWritten();
    }

    @Override
    public void markDeviceDirty() {
        nativeOps.dbTickDeviceWrite(ptrDataBuffer);
        ownerDevice = vulkanDevice(targetDevice());
    }

    void markEverywhere() {
        nativeOps.dbTickDeviceWrite(ptrDataBuffer);
        nativeOps.dbTickHostRead(ptrDataBuffer);
        ownerDevice = vulkanDevice(targetDevice());
    }
}
