/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.jcublas;

import lombok.Getter;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;

/**
 * Wraps the allocation
 * and freeing of resources on a cuda device
 * @author bam4d
 *
 */
public class CublasPointer implements AutoCloseable {

    /**
     * The underlying cuda buffer that contains the host and device memory
     */
    private JCudaBuffer buffer;
    private Pointer devicePointer;
    private Pointer hostPointer;
    @Getter
    private boolean closed = false;
    private INDArray arr;
    private CudaContext cudaContext;
    private boolean resultPointer = false;


    /**
     * frees the underlying
     * device memory allocated for this pointer
     */
    @Override
    public void close() throws Exception {
        if (!isResultPointer()) {
            destroy();
        }
    }


    /**
     * The actual destroy method
     */
    public void destroy() {

    }


    /**
     *
     * @return
     */
    public JCudaBuffer getBuffer() {
        return buffer;
    }

    /**
     *
     * @return
     */
    public Pointer getDevicePointer() {
        return devicePointer;
    }

    public Pointer getHostPointer() {
        return hostPointer;
    }

    public void setHostPointer(Pointer hostPointer) {
        this.hostPointer = hostPointer;
    }

    /**
     * Creates a CublasPointer
     * for a given JCudaBuffer
     * @param buffer
     */
    public CublasPointer(JCudaBuffer buffer, CudaContext context) {
        this.buffer = buffer;
        //        this.devicePointer = AtomicAllocator.getInstance().getPointer(new Pointer(buffer.originalDataBuffer() == null ? buffer : buffer.originalDataBuffer()), AllocationUtils.buildAllocationShape(buffer), true);
        this.cudaContext = context;
        /*
        context.initOldStream();
        
        DevicePointerInfo info = buffer.getPointersToContexts().get(Thread.currentThread().getName(), Triple.of(0, buffer.length(), 1));
        hostPointer = info.getPointers().getHostPointer();
        ContextHolder.getInstance().getMemoryStrategy().setData(devicePointer,0,1,buffer.length(),info.getPointers().getHostPointer());
        buffer.setCopied(Thread.currentThread().getName());
        */
    }

    /**
     * Creates a CublasPointer for a given INDArray.
     *
     * This wrapper makes sure that the INDArray offset, stride
     * and memory pointers are accurate to the data being copied to and from the device.
     *
     * If the copyToHost function is used in in this class,
     * the host buffer offset and data length is taken care of automatically
     * @param array
     */
    public CublasPointer(INDArray array, CudaContext context) {
        //we have to reset the pointer to be zero offset due to the fact that
        //vector based striding won't work with an array that looks like this

        this.cudaContext = context;
        this.devicePointer = AtomicAllocator.getInstance().getPointer(array, context);

        /*
        if(array instanceof IComplexNDArray) {
            if(array.length() * 2 < array.data().length()  && !array.isVector()) {
                array = Shape.toOffsetZero(array);
            }
        }
        
        buffer = (JCudaBuffer) array.data();
        
        //the opName of this thread for knowing whether to copy data or not
        //String opName = Thread.currentThread().getName();
        this.arr = array;
        if(array.elementWiseStride() < 0) {
            this.arr = array.dup();
            buffer = (JCudaBuffer) this.arr.data();
            if(this.arr.elementWiseStride() < 0)
                throw new IllegalStateException("Unable to iterate over buffer");
        }
        */
        //int compLength = arr instanceof IComplexNDArray ? arr.length() * 2 : arr.length();
        ////int stride = arr instanceof IComplexNDArray ? BlasBufferUtil.getBlasStride(arr) / 2 : BlasBufferUtil.getBlasStride(arr);
        //no striding for upload if we are using the whole buffer
        //  System.out.println("Allocation offset: ["+array.offset()+"], length: ["+compLength+"], stride: ["+ stride+"]");

        /*
                buffer.getPointer(
                this.arr,
                stride
                ,this.arr.offset()
                ,compLength);
        */


        /**
         * Neat edge case here.
         *
         * The striding will overshoot the original array
         * when the offset is zero (the case being when offset is zero
         * sayon a getRow(0) operation.
         *
         * We need to allocate the data differently here
         * due to how the striding works out.
         */
        // Copy the data to the device iff the whole buffer hasn't been copied
        /*
        
        //Data is already copied into CUDA buffer during allocation at getPointer
        
        if(!buffer.copied(opName)) {
            ContextHolder.getInstance().getMemoryStrategy().setData(buffer,0,1,buffer.length());
            //mark the buffer copied
            buffer.setCopied(opName);
        
        }*/

        /*
        DevicePointerInfo info = buffer.getPointersToContexts().get(Thread.currentThread().getName(), Triple.of(0, buffer.length(), 1));
        hostPointer = info.getPointers().getHostPointer();
        */
    }


    /**
     * Whether this is a result pointer or not
     * A result pointer means that this
     * pointer should not automatically be freed
     * but instead wait for results to accumulate
     * so they can be returned from
     * the gpu first
     * @return
     */
    public boolean isResultPointer() {
        return resultPointer;
    }

    /**
     * Sets whether this is a result pointer or not
     * A result pointer means that this
     * pointer should not automatically be freed
     * but instead wait for results to accumulate
     * so they can be returned from
     * the gpu first
     * @return
     */
    public void setResultPointer(boolean resultPointer) {
        this.resultPointer = resultPointer;
    }

    @Override
    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("NativePointer: [" + devicePointer.address() + "]");
        return sb.toString();
    }


    public static void free(CublasPointer... pointers) {
        for (CublasPointer pointer : pointers) {
            try {
                pointer.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }


}
