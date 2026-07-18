package org.nd4j.nativeblas;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.api.memory.deallocation.OpaqueNDArrayDeallocator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;

/**
 * OpaqueNDArray is a wrapper class for an opaque representation of an n-dimensional array used in ND4J.
 * It delegates most of its operations to the native backend via {@link Nd4j#getNativeOps()}.
 * This class is equivalent to an `sd::NDArray*` in the native C++ code and is used to interface with the
 * underlying C++ implementation of ND4J.
 *
 * <p>
 * It supports various operations including creation, deletion, and conversion between {@link INDArray} and
 * its native representation.
 * </p>
 *
 * <p>
 * Instances of this class should be used with care, especially in terms of resource management,
 * as it directly allocates and deallocates memory in native code.
 * </p>
 *
 * <p><b>Memory Management:</b> As of this version, OpaqueNDArray is integrated with {@link DeallocatorService}
 * for reliable memory cleanup. Previously relied on JavaCPP finalizers which were unreliable. Now uses
 * {@link OpaqueNDArrayDeallocator} for deterministic cleanup.</p>
 *
 * <p>This class extends {@link org.bytedeco.javacpp.Pointer}.</p>
 *
 * <p>Related classes include {@link OpaqueNDArrayArr}.</p>
 *
 * @see org.nd4j.linalg.api.ndarray.INDArray
 * @see org.bytedeco.javacpp.Pointer
 * @see OpaqueNDArrayArr
 * @see Nd4j#getNativeOps()
 * @see OpaqueNDArrayDeallocator
 *
 * @version 1.1
 * @since 2024.2.1
 */
@Slf4j
public class OpaqueNDArray extends Pointer {

    /**
     * Flag to enable/disable Java stack trace capture for debugging.
     * Stack trace capture is expensive and should only be enabled for debugging.
     * Set via system property "nd4j.opaque.stacktrace" (default: false).
     */
    private static final boolean CAPTURE_STACK_TRACE =
            Boolean.parseBoolean(System.getProperty(ND4JSystemProperties.OPAQUE_STACKTRACE, "false"));

    // Track the deallocator and exact backend owner for this instance.
    private OpaqueNDArrayDeallocator deallocator;
    private NativeBufferOwner backendOwner;

    private OpaqueDataBuffer shapeInfoBufferRef;
    private OpaqueDataBuffer dataBufferRef;
    private OpaqueDataBuffer specialBufferRef;

    /**
     * Constructs an OpaqueNDArray from a given Pointer.
     *
     * @param p The Pointer object representing the native memory address.
     */
    public OpaqueNDArray(Pointer p) {
        super(p);
    }

    public OpaqueNDArray attachOwner(NativeBufferOwner owner) {
        if (owner == null) {
            throw new IllegalArgumentException("NativeBufferOwner cannot be null");
        }
        if (backendOwner != null && backendOwner.nativeOps() != owner.nativeOps()) {
            if (deallocator != null || shapeInfoBufferRef != null
                    || dataBufferRef != null || specialBufferRef != null) {
                throw new IllegalStateException("OpaqueNDArray already belongs to a different backend");
            }
        }
        backendOwner = owner;
        return this;
    }

    public NativeBufferOwner backendOwner() {
        if (backendOwner == null) {
            throw new IllegalStateException(
                    "OpaqueNDArray has no backend owner; create it with an owner-scoped factory");
        }
        return backendOwner;
    }

    private NativeOps nativeOps() {
        return backendOwner().nativeOps();
    }

    /**
     * Creates an OpaqueNDArray with given buffers and offset.
     * This method delegates the creation to {@link Nd4j#getNativeOps()}.
     *
     * <p><b>Memory Management:</b> The created OpaqueNDArray is automatically registered
     * with {@link DeallocatorService} for cleanup. You can also explicitly call {@link #close()}
     * for immediate cleanup.</p>
     *
     * <p><b>Multi-GPU Support:</b> This method automatically switches to the correct device
     * context based on the buffer's device before calling native code, then restores the
     * original device context afterward.</p>
     *
     * @param shapeInfo The shape information buffer.
     * @param buffer The primary data buffer.
     * @param specialBuffer The special buffer (e.g., for GPU data).
     * @param offset The offset in the buffer.
     * @return A new OpaqueNDArray registered with DeallocatorService.
     */
    public static OpaqueNDArray create(
            OpaqueDataBuffer shapeInfo,
            OpaqueDataBuffer buffer,
            OpaqueDataBuffer specialBuffer,
            long offset) {
        return create(ownerOf(shapeInfo, buffer, specialBuffer),
                shapeInfo, buffer, specialBuffer, offset);
    }

    /**
     * Creates an OpaqueNDArray entirely through the selected backend owner.
     */
    public static OpaqueNDArray create(
            NativeBufferOwner owner,
            OpaqueDataBuffer shapeInfo,
            OpaqueDataBuffer buffer,
            OpaqueDataBuffer specialBuffer,
            long offset) {
        if (owner == null) {
            throw new IllegalArgumentException("NativeBufferOwner cannot be null");
        }
        verifyOwner(owner, shapeInfo, "shapeInfo");
        verifyOwner(owner, buffer, "buffer");
        verifyOwner(owner, specialBuffer, "specialBuffer");

        String javaStackTrace = CAPTURE_STACK_TRACE ? captureJavaStackTrace() : null;
        NativeOps ops = owner.nativeOps();
        int currentDevice = owner.currentDevice();
        int targetDevice = targetDevice(currentDevice, buffer, specialBuffer, shapeInfo);
        int deviceCount = owner.deviceCount();
        if (targetDevice < 0 || targetDevice >= deviceCount) {
            throw new IllegalArgumentException(
                    "Invalid target device " + targetDevice + " for owning backend with "
                            + deviceCount + " devices");
        }

        boolean switchedDevice = currentDevice != targetDevice;
        if (switchedDevice) {
            owner.setDevice(targetDevice);
        }

        OpaqueNDArray array;
        try {
            array = ops.create(shapeInfo, buffer, specialBuffer, offset);
            if (array != null) {
                array.retainReference();
            }
        } finally {
            if (switchedDevice) {
                owner.setDevice(currentDevice);
            }
        }

        if (array == null || array.isNull()) {
            throw new IllegalStateException("Backend failed to create OpaqueNDArray");
        }

        array.attachOwner(owner);
        try {
            array.shapeInfoBufferRef = shapeInfo;
            array.dataBufferRef = buffer;
            array.specialBufferRef = specialBuffer;
            registerWithDeallocatorService(array, owner);


            if (javaStackTrace != null && !javaStackTrace.isEmpty()) {
                ops.updateAllocationJavaStackTrace(array, javaStackTrace);
            }
        } catch (Exception e) {
            ops.deleteNDArray(array);
            array.setNull();
            throw e;
        }

        return array;
    }

    private static NativeBufferOwner ownerOf(OpaqueDataBuffer... buffers) {
        for (OpaqueDataBuffer buffer : buffers) {
            if (buffer != null && !buffer.isNull()) {
                return buffer.backendOwner();
            }
        }
        throw new IllegalArgumentException(
                "At least one live owner-scoped buffer is required to create an OpaqueNDArray");
    }

    private static void verifyOwner(NativeBufferOwner owner, OpaqueDataBuffer buffer, String role) {
        if (buffer != null && !buffer.isNull()
                && buffer.backendOwner().nativeOps() != owner.nativeOps()) {
            throw new IllegalArgumentException(role + " belongs to a different native backend");
        }
    }

    private static int targetDevice(int currentDevice, OpaqueDataBuffer... buffers) {
        for (OpaqueDataBuffer buffer : buffers) {
            if (buffer != null && !buffer.isNull()) {
                int device = buffer.deviceId();
                if (device >= 0) {
                    return device;
                }
            }
        }
        return currentDevice;
    }

    /**
     * Captures the current Java stack trace as a string.
     * This is called from Java side to get the full stack trace before JNI boundary.
     */
    private static String captureJavaStackTrace() {
        StringBuilder sb = new StringBuilder();
        StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
        // Skip first 2 frames (getStackTrace and captureJavaStackTrace)
        for (int i = 2; i < stackTrace.length && i < 64; i++) {
            sb.append("  at ").append(stackTrace[i].toString()).append("\n");
        }
        return sb.toString();
    }

    /**
     * Registers this OpaqueNDArray with the DeallocatorService for automatic cleanup.
     * This replaces reliance on unreliable JavaCPP finalizers.
     *
     * @param array The array to register
     * @throws RuntimeException if registration fails (array must be cleaned up by caller)
     */
    private static void registerWithDeallocatorService(
            OpaqueNDArray array, NativeBufferOwner owner) {
        try {
            DeallocatorService service = owner.deallocatorService();
            long uniqueId = service.nextValue();
            int targetDevice = array.deviceId();
            if (targetDevice < 0) {
                targetDevice = owner.currentDevice();
            }
            int deviceCount = owner.deviceCount();
            if (targetDevice < 0 || targetDevice >= deviceCount) {
                throw new IllegalArgumentException(
                        "Invalid array device " + targetDevice + " for owning backend with "
                                + deviceCount + " devices");
            }

            OpaqueNDArrayDeallocator deallocator = new OpaqueNDArrayDeallocator(
                    array, uniqueId, targetDevice, owner);
            array.deallocator = deallocator;
            service.pickObject(deallocator);

            if (log.isTraceEnabled()) {
                log.trace("Registered OpaqueNDArray {} with DeallocatorService on device {}",
                        uniqueId, targetDevice);
            }
        } catch (Exception e) {
            log.error("Failed to register OpaqueNDArray with its backend DeallocatorService", e);
            throw new RuntimeException("Failed to register array with DeallocatorService", e);
        }
    }

    /**
     * Gets the data type of the OpaqueNDArray.
     * This method uses {@link Nd4j#getNativeOps()} to retrieve the data type.
     *
     * @return The DataType of the array.
     */
    public DataType dataType() {
        return ArrayOptionsHelper.dataType(extras());
    }

    /**
     * Gets the extra information of the OpaqueNDArray.
     * This method uses {@link Nd4j#getNativeOps()} to retrieve the extra information.
     *
     * @return A long value representing the extra information.
     */
    public long extras() {
        return Shape.extras(shapeInfo());
    }

    /**
     * Retrieves the offset of an OpaqueNDArray.
     * This method uses {@link Nd4j#getNativeOps()} to retrieve the offset.
     *
     * @param array The OpaqueNDArray whose offset is to be retrieved.
     * @return The offset value.
     */
    public static long getOpaqueNDArrayOffset(OpaqueNDArray array) {
        return array.nativeOps().getOpaqueNDArrayOffset(array);
    }

    /**
     * Retrieves the shape information of an OpaqueNDArray.
     * This method uses {@link Nd4j#getNativeOps()} to retrieve the shape information.
     *
     * @param array The OpaqueNDArray whose shape information is to be retrieved.
     * @return An array of long values representing the shape information.
     */
    public static long[] getOpaqueNDArrayShapeInfo(OpaqueNDArray array) {
        NativeOps ops = array.nativeOps();
        LongPointer ret = ops.getOpaqueNDArrayShapeInfo(array);
        if (ret == null || ret.isNull()) return null;
        long len = ops.getShapeInfoLength(array);
        if (len <= 0) return null;
        ret.capacity(len);
        long[] retArr = new long[(int) len];
        ret.get(retArr);
        return retArr;
    }

    /**
     * Retrieves the primary buffer of an OpaqueNDArray.
     * This method uses {@link Nd4j#getNativeOps()} to retrieve the buffer.
     *
     * @param array The OpaqueNDArray whose buffer is to be retrieved.
     * @return A Pointer to the buffer.
     */
    public static Pointer getOpaqueNDArrayBuffer(OpaqueNDArray array) {
        return array.nativeOps().getOpaqueNDArrayBuffer(array).retainReference();
    }

    /**
     * Retrieves the special buffer of an OpaqueNDArray.
     * This method uses {@link Nd4j#getNativeOps()} to retrieve the special buffer.
     *
     * @param array The OpaqueNDArray whose special buffer is to be retrieved.
     * @return A Pointer to the special buffer.
     */
    public static Pointer getOpaqueNDArraySpecialBuffer(OpaqueNDArray array) {
        Pointer ptr = array.nativeOps().getOpaqueNDArraySpecialBuffer(array);
        return ptr != null ? ptr.retainReference() : null;
    }

    /**
     * Gets the length of the OpaqueNDArray.
     * This method uses {@link Nd4j#getNativeOps()} to retrieve the length.
     *
     * @param array The OpaqueNDArray whose length is to be retrieved.
     * @return The length of the array.
     */
    public static long getOpaqueNDArrayLength(OpaqueNDArray array) {
        return array.nativeOps().getOpaqueNDArrayLength(array);
    }

    /**
     * Deletes an OpaqueNDArray.
     * This method uses {@link Nd4j#getNativeOps()} to delete the array.
     *
     * @param array The OpaqueNDArray to delete.
     */
    public static void deleteNDArray(OpaqueNDArray array) {
        array.nativeOps().deleteNDArray(array);
    }

    /**
     * Deletes and nullifies an OpaqueNDArray.
     * This method uses {@link Nd4j#getNativeOps()} to delete the array.
     *
     * @param array The OpaqueNDArray to delete.
     */
    public static void delete(OpaqueNDArray array) {
        if (array != null && !array.isNull()) {
            deleteNDArray(array);
            array.setNull();
        }
    }

    /**
     * Closes the current OpaqueNDArray, releasing any allocated resources.
     * This method provides explicit cleanup and is preferred over waiting for
     * automatic cleanup via DeallocatorService.
     *
     * <p><b>Note:</b> After calling close(), this OpaqueNDArray should not be used.</p>
     */
    @Override
    public void close() {
        // During JVM shutdown, skip native deallocation to avoid calling free()
        // on potentially corrupted heap metadata. The OS reclaims all process memory on exit.
        if (DeallocatorService.getShutdownInProgress().get()) {
            return;
        }

        if (deallocator != null) {
            // Only deallocate if not already done - prevents double-free
            if (!deallocator.isDeallocated()) {
                deallocator.deallocate();
            }
            // If deallocator exists but is already deallocated, do nothing
        } else {
            // Fallback ONLY if not registered with DeallocatorService at all
            // This should only happen for OpaqueNDArrays created without registration
            delete(this);
        }

        // Clear buffer references to allow them to be freed by DeallocatorService
        // The native NDArray is already deleted at this point, so we no longer need
        // to keep these buffers alive.
        shapeInfoBufferRef = null;
        dataBufferRef = null;
        specialBufferRef = null;
    }

    /**
     * Converts an INDArray to an OpaqueNDArray without caching.
     * This method creates a new OpaqueNDArray each time it's called.
     *
     * <p><b>Important:</b> The returned OpaqueNDArray should be closed when done
     * to avoid memory leaks. Use try-with-resources pattern.</p>
     *
     * @param array The INDArray to convert.
     * @return The corresponding OpaqueNDArray.
     */
    public static OpaqueNDArray fromINDArrayUncached(INDArray array) {
        if (array == null) {
            return null;
        }

        DataBuffer buffer = array.data();
        DataBuffer shapeInfo = array.shapeInfoDataBuffer();

        boolean bufferIsEmpty = buffer == null || buffer.length() < 1;

        OpaqueNDArray opaque = create(
                shapeInfo.opaqueBuffer(),
                bufferIsEmpty ? null : buffer.opaqueBuffer(),
                bufferIsEmpty ? null : buffer.opaqueBuffer(),
                array.offset()
        );

        if (opaque != null && !array.closeable()) {
            // Only mark the OpaqueNDArray deallocator as constant to prevent GC
            // from freeing the native wrapper during op execution. Do NOT propagate
            // to the data buffer — views share their parent's DataBuffer, and marking
            // it constant would poison all D2H sync for the parent and all other views.
            if (opaque.deallocator != null) {
                opaque.deallocator.setConstant(true);
            }
        }

        return opaque;
    }

    /**
     * Converts an INDArray without consulting ND4J's primary backend.
     */
    public static OpaqueNDArray fromINDArrayUncached(
            NativeBufferOwner owner, INDArray array) {
        if (owner == null) {
            throw new IllegalArgumentException("NativeBufferOwner cannot be null");
        }
        if (array == null) {
            return null;
        }
        if (array.wasClosed()) {
            throw new IllegalStateException("Cannot wrap a closed INDArray");
        }

        DataBuffer shapeInfo = array.shapeInfoDataBuffer();
        if (shapeInfo == null || shapeInfo.wasClosed()) {
            throw new IllegalStateException("Cannot wrap an INDArray without live shape information");
        }

        DataBuffer buffer = array.data();
        boolean arrayEmpty = array.isEmpty();
        if (buffer == null) {
            if (!arrayEmpty) {
                throw new IllegalStateException(
                        "Cannot wrap a non-empty INDArray without a data buffer");
            }
        } else {
            if (buffer.wasClosed()) {
                throw new IllegalStateException("Cannot wrap an INDArray with a closed data buffer");
            }
            if (!arrayEmpty && buffer.length() < 1) {
                throw new IllegalStateException(
                        "Cannot wrap a non-empty INDArray with an empty data buffer");
            }
        }

        OpaqueDataBuffer shapeOpaque = shapeInfo.opaqueBuffer();
        OpaqueDataBuffer dataOpaque = arrayEmpty ? null : buffer.opaqueBuffer();
        return create(owner, shapeOpaque, dataOpaque, dataOpaque, array.offset());
    }

    /**
     * Converts an INDArray to an OpaqueNDArray.
     * This method uses caching via {@link INDArray#getOrCreateOpaqueNDArray()}.
     *
     * <p><b>Note:</b> The cached OpaqueNDArray will be cleaned up when the INDArray
     * is closed or garbage collected.</p>
     *
     * @param array The INDArray to convert.
     * @return The corresponding OpaqueNDArray (may be cached).
     */
    public static OpaqueNDArray fromINDArray(INDArray array) {
        return fromINDArray(array, true);
    }

    /**
     * Converts an INDArray to an OpaqueNDArray without forcing host-to-device
     * synchronization. Callers that use this path must prepare inputs through
     * native execution ownership, for example NDArray::prepareSpecialUse.
     *
     * @param array The INDArray to convert.
     * @return The corresponding OpaqueNDArray (may be cached).
     */
    public static OpaqueNDArray fromINDArrayNoSync(INDArray array) {
        return fromINDArray(array, false);
    }

    public static OpaqueNDArray fromINDArray(NativeBufferOwner owner, INDArray array) {
        return fromINDArray(owner, array, true);
    }

    public static OpaqueNDArray fromINDArrayNoSync(
            NativeBufferOwner owner, INDArray array) {
        return fromINDArray(owner, array, false);
    }

    private static OpaqueNDArray fromINDArray(
            NativeBufferOwner owner, INDArray array, boolean syncHostToDevice) {
        if (array == null) {
            return null;
        }
        if (array.wasClosed()) {
            throw new IllegalStateException("Cannot wrap a closed INDArray");
        }
        DataBuffer buffer = array.data();
        if (buffer != null && !buffer.wasClosed() && syncHostToDevice && !array.isEmpty()) {
            OpaqueDataBuffer opaqueBuffer = buffer.opaqueBuffer();
            verifyOwner(owner, opaqueBuffer, "data");
            opaqueBuffer.syncToSpecial();
        }
        return fromINDArrayUncached(owner, array);
    }

    private static OpaqueNDArray fromINDArray(INDArray array, boolean syncHostToDevice) {
        if(array == null) {
            return null;
        }

        // Guard against arrays with closed/null data buffers.
        // Control flow dead branches (Switch/Merge) may produce arrays whose
        // DataBuffers have been freed by session cleanup, or never allocated.
        // Wrapping such arrays creates OpaqueNDArrays with null PointerWrapper
        // internals, causing SIGSEGV in native code.
        if (array.wasClosed()) {
            throw new IllegalStateException(
                "Cannot create OpaqueNDArray from closed INDArray (id=" + array.getId() + ").");
        }
        if (array.data() == null || array.data().wasClosed()) {
            // Array has no live data — either never allocated (control flow intermediates)
            // or DataBuffer was freed (dead branch cleanup).
            DataBuffer shapeInfo = array.shapeInfoDataBuffer();
            if (shapeInfo == null || shapeInfo.wasClosed()) {
                // Can't even read shape — create a minimal scalar placeholder
                return fromINDArrayUncached(Nd4j.scalar(0.0f));
            }
            // Check if the array is genuinely empty (not just a dead control-flow branch).
            // We must check BOTH the Java-side isEmpty() AND the native shape info EMPTY bit,
            // because Nd4j.empty(DataType) singletons store the EMPTY bit in javaShapeInformation
            // but the native shape DataBuffer may only have the dtype bit (no EMPTY bit).
            // In that case array.isEmpty() (which uses javaShapeInformation) returns true but
            // Shape.isEmpty(shapeInfo.asLong()) returns false. Both must be checked.
            boolean shapeInfoEmpty = Shape.isEmpty(shapeInfo.asLong());
            // Also check Java-side isEmpty() which uses javaShapeInformation (has EMPTY bit for Nd4j.empty() singletons).
            boolean javaEmpty = array.isEmpty();
            if (shapeInfoEmpty) {
                // Native shape info has EMPTY bit set — safe to pass null buffer to C++.
                return fromINDArrayUncached(array);
            }
            if (javaEmpty) {
                // Java says empty but native shape info does NOT have EMPTY bit.
                // C++ createOpaqueNDArray checks: rank==0 && buffer==nullptr for "javaStyleEmpty".
                // If native shape has rank > 0 or non-zero length, C++ throws
                // "not empty but null buffer". Only pass through when native shape is rank-0.
                long rank = shapeInfo.asLong()[0];
                long nativeLength = Shape.length(shapeInfo.asLong());
                if (rank == 0 || nativeLength == 0) {
                    // Rank-0 or zero-length: C++ javaStyleEmpty path handles this safely.
                    return fromINDArrayUncached(array);
                }
                // Java EMPTY bit is set but native shape has rank>0 and non-zero length.
                // This is a mismatch — treat as non-empty and create zero-filled replacement.
                INDArray replacement = Nd4j.zeros(array.dataType(), array.shape());
                return fromINDArrayUncached(replacement);
            }
            // Non-empty shape but null data (dead control flow branches, freed buffers).
            // Create a zero-filled replacement to avoid native "not empty but null buffer" error.
            INDArray replacement = Nd4j.zeros(array.dataType(), array.shape());
            return fromINDArrayUncached(replacement);
        }

        // Sync host→device before passing to native ops.
        // Java-side operations like putScalar() write to the HOST buffer and mark it as
        // authoritative. Native CUDA ops read from the DEVICE buffer. Without this sync,
        // the device buffer may contain stale data, causing operations like maxNumber()
        // to return incorrect results after putScalar() or put() modifications.
        // On CPU backend, dbSyncToSpecial is a no-op.
        if (syncHostToDevice && !array.isEmpty() && array.data() != null && !array.data().wasClosed()) {
            OpaqueDataBuffer opaqueBuffer = array.data().opaqueBuffer();
            if (opaqueBuffer != null && !opaqueBuffer.isNull()) {
                // Sync host→device for arrays where Java-side writes may have modified host data.
                // Use non-force sync so actuality counters are respected: if a CUDA kernel already
                // wrote to the device buffer (device is authoritative), the sync is skipped.
                // Previously, INT/LONG arrays used forced sync which overwrote correct device data
                // (e.g., shape_of kernel output) with stale host zeros.
                opaqueBuffer.syncToSpecial();
            }
        }

        OpaqueNDArray opaque = array.getOrCreateOpaqueNDArray();
        if (opaque == null || opaque.isNull()) {
            throw new IllegalStateException(
                "Failed to create OpaqueNDArray from INDArray (id=" + array.getId() +
                ", closed=" + array.wasClosed() +
                "). The native pointer is null. This indicates premature deallocation.");
        }

        if (!array.closeable() && !opaque.isConstant()) {
            // Only mark the OpaqueNDArray deallocator as constant to prevent GC
            // from freeing the native wrapper during op execution. Do NOT propagate
            // to the data buffer — views share their parent's DataBuffer, and marking
            // it constant would poison all D2H sync for the parent and all other views.
            // See OpaqueNDArray.create() for the careful constant-propagation logic.
            if (opaque.deallocator != null) {
                opaque.deallocator.setConstant(true);
            }
        }

        return opaque;
    }

    /**
     * Converts an OpaqueNDArray to an INDArray.
     * This method uses the data and shape information from {@link Nd4j#getNativeOps()} to create the INDArray.
     *
     * @param opaqueArray The OpaqueNDArray to convert.
     * @return The corresponding INDArray.
     */
    public static INDArray toINDArray(OpaqueNDArray opaqueArray) {
        if (opaqueArray == null || opaqueArray.isNull()) {
            return null;
        }

        long offset = opaqueArray.getOffset();
        long[] shapeInfoPtr = opaqueArray.shapeInfo();
        Pointer bufferPtr = opaqueArray.buffer();
        Pointer specialBufferPtr = opaqueArray.specialBuffer();

        long length = opaqueArray.length();

        // Extract shape information
        long[] shape = Shape.shape(shapeInfoPtr);
        long[] stride = Shape.stride(shapeInfoPtr);
        char order = Shape.order(shapeInfoPtr);
        long ews = Shape.elementWiseStride(shapeInfoPtr);
        long extras = Shape.extras(shapeInfoPtr);

        // Create LongShapeDescriptor
        LongShapeDescriptor descriptor = LongShapeDescriptor.builder()
                .shape(shape)
                .stride(stride)
                .offset(offset)
                .ews(ews)
                .order(order)
                .extras(extras)
                .build();

        // Create DataBuffer from the OpaqueNDArray's buffer
        DataType dataType = ArrayOptionsHelper.dataType(extras);
        DataBuffer buffer = Nd4j.createBuffer(bufferPtr, specialBufferPtr, length, dataType);
        
        // Create INDArray using the descriptor and buffer
        return Nd4j.create(buffer, descriptor);
    }

    // Convenience methods

    /**
     * Returns the device id based on the retained OpaqueDataBuffer references.
     * Falls back to -1 if no buffer references are available.
     *
     * @return device id for this array, or -1 if unknown
     */
    public int deviceId() {
        if (dataBufferRef != null && !dataBufferRef.isNull()) {
            return dataBufferRef.deviceId();
        }
        if (specialBufferRef != null && !specialBufferRef.isNull()) {
            return specialBufferRef.deviceId();
        }
        if (shapeInfoBufferRef != null && !shapeInfoBufferRef.isNull()) {
            return shapeInfoBufferRef.deviceId();
        }
        return -1;
    }

    /**
     * Gets the offset of the current OpaqueNDArray.
     *
     * @return The offset of the array.
     */
    public long getOffset() {
        return getOpaqueNDArrayOffset(this);
    }

    /**
     * Gets the shape information of the current OpaqueNDArray.
     *
     * @return An array of long values representing the shape information.
     */
    public long[] shapeInfo() {
        return getOpaqueNDArrayShapeInfo(this);
    }

    /**
     * Gets the primary buffer of the current OpaqueNDArray.
     *
     * @return A Pointer to the buffer.
     */
    public Pointer buffer() {
        return getOpaqueNDArrayBuffer(this);
    }

    /**
     * Gets the special buffer of the current OpaqueNDArray.
     *
     * @return A Pointer to the special buffer.
     */
    public Pointer specialBuffer() {
        return getOpaqueNDArraySpecialBuffer(this);
    }

    /**
     * Gets the length of the current OpaqueNDArray.
     *
     * @return The length of the array.
     */
    public long length() {
        return getOpaqueNDArrayLength(this);
    }

    /**
     * Gets the deallocator associated with this OpaqueNDArray.
     *
     * @return The deallocator or null if not registered
     */
    public OpaqueNDArrayDeallocator getDeallocator() {
        return deallocator;
    }

    /**
     * Gets the data buffer reference held by this OpaqueNDArray.
     * Used by the deallocator to check if underlying buffers are constant.
     *
     * @return The data buffer reference or null
     */
    public OpaqueDataBuffer getDataBufferRef() {
        return dataBufferRef;
    }

    /**
     * Gets the shape info buffer reference held by this OpaqueNDArray.
     * Used by the deallocator to check if underlying buffers are constant.
     *
     * @return The shape info buffer reference or null
     */
    public OpaqueDataBuffer getShapeInfoBufferRef() {
        return shapeInfoBufferRef;
    }

    /**
     * Gets the special buffer reference held by this OpaqueNDArray.
     * Used by the deallocator to keep the buffer alive during NDArray cleanup.
     *
     * @return The special buffer reference or null
     */
    public OpaqueDataBuffer getSpecialBufferRef() {
        return specialBufferRef;
    }

    /**
     * Marks this OpaqueNDArray as constant (immutable).
     * Constant arrays are never freed by the DeallocatorService because they
     * wrap cached/shared data that has a different lifecycle.
     *
     * This should be called for arrays created from constant INDArrays
     * (e.g., model weights, constants in SameDiff).
     *
     * @param isConstant true to mark as constant, false otherwise
     */
    public void setConstant(boolean isConstant) {
        if (deallocator != null) {
            deallocator.setConstant(isConstant);
        }

        if (isConstant) {
            if (dataBufferRef != null) {
                dataBufferRef.setConstant(true);
            }
            if (shapeInfoBufferRef != null) {
                shapeInfoBufferRef.setConstant(true);
            }
            if (specialBufferRef != null) {
                specialBufferRef.setConstant(true);
            }
        }
    }

    /**
     * Returns whether this OpaqueNDArray is marked as constant.
     *
     * @return true if constant, false otherwise
     */
    public boolean isConstant() {
        return deallocator != null && deallocator.isConstant();
    }
}
