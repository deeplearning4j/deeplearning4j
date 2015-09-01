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
import jcuda.jcublas.JCublas;
import jcuda.jcublas.JCublas2;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Arrays;

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
    final JCudaBuffer buffer;
    final Pointer devicePointer;
    private boolean closed = false;
    private INDArray arr;


    /**
     * frees the underlying
     * device memory allocated for this pointer
     */
    @Override
    public void close() throws Exception {
        if(!closed) {
            if(arr != null)
                buffer.freeDevicePointer(arr.offset(),arr.length());
            else
                buffer.freeDevicePointer(0,buffer.length());
            closed = true;
        }
    }

    public JCudaBuffer getBuffer() {
        return buffer;
    }

    public Pointer getDevicePointer() {
        return devicePointer;
    }

    /**
     * copies the result to the host buffer
     */
    public void copyToHost() {
        if(arr != null) {
            int compLength = arr instanceof IComplexNDArray ? arr.length() * 2 : arr.length();
            buffer.copyToHost(arr.offset(),compLength);
        }
        else {
            buffer.copyToHost(0,buffer.length());
        }
    }


    /**
     * Creates a CublasPointer
     * for a given JCudaBuffer
     * @param buffer
     */
    public CublasPointer(JCudaBuffer buffer) {
        this.buffer = buffer;
        this.devicePointer = buffer.getDevicePointer(1, 0, buffer.length());
        // Copy the data to the device
        JCublas2.cublasSetVectorAsync(
                buffer.length()
                , buffer.getElementSize()
                , buffer.getHostPointer()
                , 1
                , devicePointer
                , 1
                , ContextHolder.getInstance().getCudaStream());
        ContextHolder.syncStream();
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
    public CublasPointer(INDArray array) {
        //we have to reset the pointer to be zero offset due to the fact that
        //vector based striding won't work with an array that looks like this
        if(array instanceof IComplexNDArray) {
            if(array.length() * 2 < array.data().length()  && !array.isVector()) {
                array = Shape.toOffsetZero(array);

            }
        }

        else if(array.length() < array.data().length() && !array.isVector())
            array = Shape.toOffsetZero(array);

        buffer = (JCudaBuffer) array.data();

        //the name of this thread for knowing whether to copy data or not
        String name = Thread.currentThread().getName();
        this.arr = array;
        int compLength = arr instanceof IComplexNDArray ? arr.length() * 2 : arr.length();
        int stride = arr instanceof IComplexNDArray ? BlasBufferUtil.getBlasStride(arr) / 2 : BlasBufferUtil.getBlasStride(arr);
        //no striding for upload if we are using the whole buffer

        this.devicePointer = buffer
                .getDevicePointer(
                        array,
                        stride
                        ,array.offset()
                        ,compLength);

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
        // if(!buffer.copied(name)) {
        JCublas.cublasSetVectorAsync(
                buffer.length()
                , array.data().getElementSize()
                , buffer.getHostPointer()
                , 1
                , buffer.getPointersToContexts().get(name, new Pair<>(0,buffer.length())).getPointer()
                , 1
                , ContextHolder.getInstance().getCudaStream());
        //mark the buffer copied
        buffer.setCopied(name);

        // }
    }




    public double[] asDoubleBuffer() {
        return buffer.asDouble();
    }

    public float[] getFloatBuffer() {
        return buffer.asFloat();
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
                    JCublas2.cublasGetVectorAsync(
                            buffer.length()
                            , buffer.getElementSize()
                            , devicePointer
                            , 1
                            , Pointer.to(set)
                            , 1
                            , ContextHolder.getInstance().getCudaStream());
                    sb.append(Arrays.toString(set));
                }
                else {
                    float[] set = new float[buffer.length()];
                    JCublas2.cublasGetVectorAsync(
                            buffer.length()
                            , buffer.getElementSize()
                            , devicePointer
                            , 1
                            , Pointer.to(set)
                            , 1
                            , ContextHolder.getInstance().getCudaStream());
                    sb.append(Arrays.toString(set));
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
            JCublas2.cublasGetVectorAsync(
                    length
                    , buffer.getElementSize()
                    ,devicePointer
                    ,BlasBufferUtil.getBlasStride(arr)
                    ,Pointer.to(set)
                    ,1
                    , ContextHolder.getInstance().getCudaStream());
            ContextHolder.syncStream();
            sb.append(Arrays.toString(set));
        }
        else {
            float[] set = new float[length];
            JCublas2.cublasGetVectorAsync(
                    length
                    , buffer.getElementSize()
                    , devicePointer
                    , BlasBufferUtil.getBlasStride(arr)
                    , Pointer.to(set)
                    , 1, ContextHolder.getInstance().getCudaStream());
            ContextHolder.syncStream();
            sb.append(Arrays.toString(set));
        }
    }

    private void appendWhereArrayLengthEqualsBufferLength(StringBuffer sb) {
        int length = arr instanceof  IComplexNDArray ? arr.length() * 2 : arr.length();
        if(arr.data().dataType() == DataBuffer.Type.DOUBLE) {
            double[] set = new double[length];
            JCublas2.cublasGetVectorAsync(
                    length
                    , buffer.getElementSize()
                    ,devicePointer
                    ,1
                    ,Pointer.to(set)
                    ,1
                    , ContextHolder.getInstance().getCudaStream());
            ContextHolder.syncStream();
            sb.append(Arrays.toString(set));
        }
        else {
            float[] set = new float[length];
            JCublas2.cublasGetVectorAsync(
                    length
                    , buffer.getElementSize()
                    ,devicePointer
                    ,1
                    ,Pointer.to(set)
                    ,1
                    , ContextHolder.getInstance().getCudaStream());
            ContextHolder.syncStream();
            sb.append(Arrays.toString(set));
        }
    }


}