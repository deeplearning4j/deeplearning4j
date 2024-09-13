package org.nd4j.nativeblas;

import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.OpaqueDataBuffer;

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
 * <p>This class extends {@link org.bytedeco.javacpp.Pointer}.</p>
 *
 * <p>Related classes include {@link OpaqueNDArrayArr}.</p>
 *
 * @see org.nd4j.linalg.api.ndarray.INDArray
 * @see org.bytedeco.javacpp.Pointer
 * @see OpaqueNDArrayArr
 * @see Nd4j#getNativeOps()
 * @see sd::NDArray*
 *
 * @version 1.0
 * @since 2024.2.1
 */
public class OpaqueNDArray extends Pointer {

    /**
     * Constructs an OpaqueNDArray from a given Pointer.
     *
     * @param p The Pointer object representing the native memory address.
     */
    public OpaqueNDArray(Pointer p) { super(p); }

    /**
     * Creates an OpaqueNDArray with given buffers and offset.
     * This method delegates the creation to {@link Nd4j#getNativeOps()}.
     *
     * @param shapeInfo The shape information buffer.
     * @param buffer The primary data buffer.
     * @param specialBuffer The special buffer (e.g., for GPU data).
     * @param offset The offset in the buffer.
     * @return A new OpaqueNDArray.
     */
    public static OpaqueNDArray create(
            OpaqueDataBuffer shapeInfo,
            OpaqueDataBuffer buffer,
            OpaqueDataBuffer specialBuffer,
            long offset) {
        return Nd4j.getNativeOps().create(shapeInfo, buffer, specialBuffer, offset);
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
        return Nd4j.getNativeOps().getOpaqueNDArrayOffset(array);
    }

    /**
     * Retrieves the shape information of an OpaqueNDArray.
     * This method uses {@link Nd4j#getNativeOps()} to retrieve the shape information.
     *
     * @param array The OpaqueNDArray whose shape information is to be retrieved.
     * @return An array of long values representing the shape information.
     */
    public static long[] getOpaqueNDArrayShapeInfo(OpaqueNDArray array) {
        LongPointer ret =  Nd4j.getNativeOps().getOpaqueNDArrayShapeInfo(array);
        long len = Nd4j.getNativeOps().getShapeInfoLength(array);
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
        return Nd4j.getNativeOps().getOpaqueNDArrayBuffer(array);
    }

    /**
     * Retrieves the special buffer of an OpaqueNDArray.
     * This method uses {@link Nd4j#getNativeOps()} to retrieve the special buffer.
     *
     * @param array The OpaqueNDArray whose special buffer is to be retrieved.
     * @return A Pointer to the special buffer.
     */
    public static Pointer getOpaqueNDArraySpecialBuffer(OpaqueNDArray array) {
        return Nd4j.getNativeOps().getOpaqueNDArraySpecialBuffer(array);
    }

    /**
     * Gets the length of the OpaqueNDArray.
     * This method uses {@link Nd4j#getNativeOps()} to retrieve the length.
     *
     * @param array The OpaqueNDArray whose length is to be retrieved.
     * @return The length of the array.
     */
    public static long getOpaqueNDArrayLength(OpaqueNDArray array) {
        return Nd4j.getNativeOps().getOpaqueNDArrayLength(array);
    }

    /**
     * Deletes an OpaqueNDArray.
     * This method uses {@link Nd4j#getNativeOps()} to delete the array.
     *
     * @param array The OpaqueNDArray to delete.
     */
    public static void deleteNDArray(OpaqueNDArray array) {
        Nd4j.getNativeOps().deleteNDArray(array);
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
     * This method is called automatically on object finalization.
     */
    @Override
    public void close() {
        delete(this);
    }

    /**
     * Converts an INDArray to an OpaqueNDArray.
     * This method uses {@link Nd4j#getNativeOps()} to create the OpaqueNDArray.
     *
     * @param array The INDArray to convert.
     * @return The corresponding OpaqueNDArray.
     */
    public static OpaqueNDArray fromINDArray(INDArray array) {
        if (array == null) {
            return null;
        }

        DataBuffer buffer = array.data();
        DataBuffer shapeInfo = array.shapeInfoDataBuffer();

        return create(
                shapeInfo.opaqueBuffer(),
                array.isEmpty() ? null : buffer.opaqueBuffer(),
                array.isEmpty() ? null : array.data().opaqueBuffer(),
                array.offset()
        );
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
        DataBuffer buffer = Nd4j.createBuffer(bufferPtr,specialBufferPtr,length,dataType);
        // Create INDArray using the descriptor and buffer
        return Nd4j.create(buffer, descriptor);
    }

    // Convenience methods

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
}