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
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.ContextHolder;

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
            buffer.freeDevicePointer();
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
        buffer.copyToHost();
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
        buffer = (JCudaBuffer) array.data();
        this.arr = array;
        //no striding for upload if we are using the whole buffer
        int deviceStride;
        int compLength = array instanceof IComplexNDArray ? array.length() * 2 : array.length();
        if(compLength == buffer.length()) {
              deviceStride = 1;
        }
        else
             deviceStride = arr.majorStride();

        if(array instanceof IComplexNDArray) {

            IComplexNDArray arr2 = (IComplexNDArray) array;
            this.devicePointer = buffer
                    .getDevicePointer(array.majorStride(),array.offset(),2 * array.length());

            // Copy the data to the device
            JCublas2.cublasSetVectorAsync(
                    array instanceof IComplexNDArray ? array.length() * 2 : array.length()
                    , array.data().getElementSize()
                    , buffer.getHostPointer().withByteOffset(arr2.blasOffset() * array.data().getElementSize())
                    , deviceStride
                    , devicePointer
                    , 1
                    , ContextHolder.getInstance().getCudaStream());

        }
        else {
            this.devicePointer = buffer
                    .getDevicePointer(
                            array.majorStride()
                            ,array.offset()
                            ,array.length());

            Pointer hostPointer = buffer.getHostPointer().withByteOffset(buffer.getElementSize() * arr.offset());


            // Copy the data to the device
            JCublas.cublasSetVectorAsync(
                    array.length()
                    , array.data().getElementSize()
                    , hostPointer
                    , deviceStride
                    , devicePointer
                    , 1
                    , ContextHolder.getInstance().getCudaStream());


        }



    }


    public boolean isClosed() {
        return closed;
    }

    @Override
    public String toString() {
        StringBuffer sb = new StringBuffer();
        if(devicePointer != null) {
            if(arr != null) {
                if(arr.data().dataType() == DataBuffer.Type.DOUBLE) {
                    double[] set = new double[arr.length()];
                    JCublas2.cublasGetVectorAsync(
                            arr.length()
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
                    float[] set = new float[arr.length()];
                    JCublas2.cublasGetVectorAsync(
                            arr.length()
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
            else {
                double[] set = new double[buffer.length()];
                JCublas2.cublasGetVectorAsync(arr.length(), buffer.getElementSize(), devicePointer, 1, Pointer.to(set), 1, ContextHolder.getInstance().getCudaStream());
                sb.append(Arrays.toString(set));

            }
        }
        else
            sb.append("No device pointer yet");
        return sb.toString();
    }


}