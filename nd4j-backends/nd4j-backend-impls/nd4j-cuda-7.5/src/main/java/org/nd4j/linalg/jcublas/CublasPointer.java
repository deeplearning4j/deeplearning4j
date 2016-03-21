/*
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

import jcuda.Pointer;
import lombok.Getter;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.context.CudaContext;

/**
 * Wraps the allocation
 * and freeing of resources on a cuda device
 * @author bam4d
 *
 */
public class CublasPointer  implements AutoCloseable {

    /**
     * The underlying cuda buffer that contains the host and device memory
     */
    private JCudaBuffer buffer;
    private Pointer devicePointer;
    private Pointer hostPointer;
    @Getter private boolean closed = false;
    private INDArray arr;
    private CudaContext cudaContext;
    private boolean resultPointer = false;


    /**
     * frees the underlying
     * device memory allocated for this pointer
     */
    @Override
    public void close() throws Exception {
        if( !isResultPointer()) {
            destroy();
        }
    }


    /**
     * The actual destroy method
     */
    public void destroy() {
        if(!closed) {
            if(arr != null) {
                buffer.freeDevicePointer(arr.offset(), arr.length(), BlasBufferUtil.getBlasStride(this.arr));
            } else {
                buffer.freeDevicePointer(0, buffer.length(),1);
            }
            closed = true;
        }
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
     * copies the result to the host buffer
     *
     *
     */
    @Deprecated
    public void copyToHost() {

        if (1 > 0) return;

        if(arr != null) {
            int compLength = arr instanceof IComplexNDArray ? arr.length() * 2 : arr.length();
            ContextHolder.getInstance().getMemoryStrategy().copyToHost(buffer,arr.offset(),arr.elementWiseStride(),compLength,cudaContext,arr.offset(),arr.elementWiseStride());
        }
        else {
            ContextHolder.getInstance().getMemoryStrategy().copyToHost(buffer,0,cudaContext);
        }
    }


    /**
     * Creates a CublasPointer
     * for a given JCudaBuffer
     * @param buffer
     */
    public CublasPointer(JCudaBuffer buffer,CudaContext context) {
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
    public CublasPointer(INDArray array,CudaContext context) {
        //we have to reset the pointer to be zero offset due to the fact that
        //vector based striding won't work with an array that looks like this
        if(array instanceof IComplexNDArray) {
            if(array.length() * 2 < array.data().length()  && !array.isVector()) {
                array = Shape.toOffsetZero(array);
            }
        }
        this.cudaContext = context;
        buffer = (JCudaBuffer) array.data();

        //the name of this thread for knowing whether to copy data or not
        //String name = Thread.currentThread().getName();
        this.arr = array;
        if(array.elementWiseStride() < 0) {
            this.arr = array.dup();
            buffer = (JCudaBuffer) this.arr.data();
            if(this.arr.elementWiseStride() < 0)
                throw new IllegalStateException("Unable to iterate over buffer");
        }

        //int compLength = arr instanceof IComplexNDArray ? arr.length() * 2 : arr.length();
        ////int stride = arr instanceof IComplexNDArray ? BlasBufferUtil.getBlasStride(arr) / 2 : BlasBufferUtil.getBlasStride(arr);
        //no striding for upload if we are using the whole buffer
      //  System.out.println("Allocation offset: ["+array.offset()+"], length: ["+compLength+"], stride: ["+ stride+"]");
        this.devicePointer = new Pointer(AtomicAllocator.getInstance().getPointer(array).address());
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

        if(!buffer.copied(name)) {
            ContextHolder.getInstance().getMemoryStrategy().setData(buffer,0,1,buffer.length());
            //mark the buffer copied
            buffer.setCopied(name);

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
        if(devicePointer != null) {
            if(arr != null) {
                if(arr instanceof IComplexNDArray
                        && arr.length() * 2
                        == buffer.length()
                        || arr.length() == buffer.length())
                    appendWhereArrayLengthEqualsBufferLength(sb);
                else
                    appendWhereArrayLengthLessThanBufferLength(sb);

            }
            else {
                if(buffer.dataType() == DataBuffer.Type.DOUBLE) {
                    double[] set = new double[buffer.length()];
                    DataBuffer setBuffer = Nd4j.createBuffer(set);
                    ContextHolder.getInstance().getMemoryStrategy().getData(setBuffer, 0, 1, buffer.length(), buffer, cudaContext, 1,0);
                    sb.append(setBuffer);
                }
                else if(buffer.dataType() == DataBuffer.Type.INT) {
                    int[] set = new int[buffer.length()];
                    DataBuffer setBuffer = Nd4j.createBuffer(set);
                    ContextHolder.getInstance().getMemoryStrategy().getData(setBuffer, 0, 1, buffer.length(),buffer, cudaContext, 1, 0);
                    sb.append(setBuffer);
                }
                else {
                    float[] set = new float[buffer.length()];
                    DataBuffer setBuffer = Nd4j.createBuffer(set);
                    ContextHolder.getInstance().getMemoryStrategy().getData(setBuffer,0,1,buffer.length(), buffer,cudaContext,1, 0);
                    sb.append(setBuffer);
                }


            }
        }
        else
            sb.append("No device pointer yet");
        return sb.toString();
    }


    private void appendWhereArrayLengthLessThanBufferLength(StringBuffer sb) {
        int length = arr instanceof  IComplexNDArray ? arr.length() * 2 : arr.length();

        if(arr.data().dataType() == DataBuffer.Type.DOUBLE) {
            double[] set = new double[length];
            DataBuffer setString = Nd4j.createBuffer(set);
            ContextHolder.getInstance().getMemoryStrategy().getData(setString, 0, 1, length,buffer, cudaContext, arr.elementWiseStride(),arr.offset());
            sb.append(setString);
        }
        else if(arr.data().dataType() == DataBuffer.Type.INT) {
            int[] set = new int[length];
            DataBuffer setString = Nd4j.createBuffer(set);
            ContextHolder.getInstance().getMemoryStrategy().getData(setString, 0, 1, length, buffer, cudaContext, arr.elementWiseStride(),arr.offset());
            sb.append(setString);
        }
        else {
            float[] set = new float[length];
            DataBuffer setString = Nd4j.createBuffer(set);
            ContextHolder.getInstance().getMemoryStrategy().getData(setString, 0, 1,length,buffer, cudaContext, arr.elementWiseStride(),arr.offset());
            sb.append(setString);
        }
    }

    private void appendWhereArrayLengthEqualsBufferLength(StringBuffer sb) {
        int length = arr instanceof  IComplexNDArray ? arr.length() * 2 : arr.length();
        if(arr.data().dataType() == DataBuffer.Type.DOUBLE) {
            double[] set = new double[length];
            DataBuffer setString = Nd4j.createBuffer(set);
            ContextHolder.getInstance().getMemoryStrategy().getData(setString,0,1,length,buffer,cudaContext,1,0);
            sb.append(setString);
        }
        else if(arr.data().dataType() == DataBuffer.Type.INT) {
            int[] set = new int[length];
            DataBuffer setString = Nd4j.createBuffer(set);
            ContextHolder.getInstance().getMemoryStrategy().getData(setString, 0, 1, length, buffer, cudaContext, 1, 0);
            sb.append(setString);
        }
        else {
            float[] set = new float[length];
            DataBuffer setString = Nd4j.createBuffer(set);
            ContextHolder.getInstance().getMemoryStrategy().getData(setString, 0, 1, length, buffer, cudaContext, 1, 0);
            sb.append(setString);
        }
    }


    public static void free(CublasPointer...pointers) {
        for(CublasPointer pointer : pointers) {
            try {
                pointer.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }


}