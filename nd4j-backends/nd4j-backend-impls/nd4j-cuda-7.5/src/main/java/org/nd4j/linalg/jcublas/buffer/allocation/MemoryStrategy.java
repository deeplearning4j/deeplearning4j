package org.nd4j.linalg.jcublas.buffer.allocation;


import jcuda.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;

/**
 *
 * An allocation strategy handles allocating
 * and freeing memory for the gpu
 * (usually relative to the compute capabilities of the gpu)
 *
 * @author Adam Gibson
 */
public interface MemoryStrategy {
    /**
     * Set the data for the buffer
     * @param buffer the buffer to set
     * @param offset the offset to start at
     * @param stride the stride to sue
     * @param length the length to go till
     * @param get
     * @param ctx
     * @param getStride
     * @param getOffset
     */
    void getData(DataBuffer buffer, int offset, int stride, int length, DataBuffer get, CudaContext ctx, int getStride, int getOffset);

    /**
     * @param buffer
     * @param offset
     * @param get
     * @param ctx
     */
    void getData(DataBuffer buffer, int offset, DataBuffer get, CudaContext ctx);

    /**
     *
     * @param buffer
     * @param offset
     * @param stride
     * @param length
     * @param hostPointer
     */
    void setData(Pointer buffer, int offset, int stride, int length, Pointer hostPointer);




    /**
     * Set the data for the buffer
     * @param buffer the buffer to set
     * @param offset the offset to start at
     * @param stride the stride to sue
     * @param length the length to go till
     */
    void setData(DataBuffer buffer, int offset, int stride, int length);

    /**
     *
     * @param buffer
     * @param offset
     */
    void setData(DataBuffer buffer, int offset);

    /**
     * Copy data to native or gpu
     * @param copy the buffer to copy
     * @return a pointer representing
     * the copied data
     */
    Object copyToHost(DataBuffer copy, int offset, CudaContext context);

    /**
     * Copy data to native or gpu
     * @param copy the buffer to copy
     * @param length
     * @param hostOffset
     *@param hostStride @return a pointer representing
     * the copied data
     */
    Object copyToHost(DataBuffer copy, int offset, int stride, int length, CudaContext context, int hostOffset, int hostStride);

    /**
     * Allocate memory for the given buffer
     * @param buffer the buffer to allocate for
     * @param stride the stride
     * @param offset the offset used for the buffer
 *               on allocation
     * @param length length
     * @param initData
     */
    Object alloc(DataBuffer buffer, int stride, int offset, int length, boolean initData);

    /**
     * Free the buffer wrt the
     * allocation strategy
     * @param buffer the buffer to free
     * @param offset the offset to free
     * @param length the length to free
     */
    void free(DataBuffer buffer, int offset, int length);

    /**
     * Validates data present in a data buffer
     * @param buffer the buffer to validate
     * @param context context used for copying
     */
    void validate(DataBuffer buffer, CudaContext context) throws Exception;

}
