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

package org.nd4j.linalg.jcublas;

import lombok.Getter;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import lombok.extern.slf4j.Slf4j;

/**
 * Wraps the allocation
 * and freeing of resources on a cuda device
 * @author bam4d
 *
 */
@Slf4j
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
     * If the copyToHost function is used in this class,
     * the host buffer offset and data length is taken care of automatically
     * @param array
     */
    public CublasPointer(INDArray array, CudaContext context) {
        //we have to reset the pointer to be zero offset due to the fact that
        //vector based striding won't work with an array that looks like this

        this.cudaContext = context;
        this.devicePointer = AtomicAllocator.getInstance().getPointer(array, context);

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
                log.error("",e);
            }
        }
    }


}
