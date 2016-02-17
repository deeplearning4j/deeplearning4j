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

package org.nd4j.linalg.jcublas.util;

import com.google.common.collect.Multimap;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.buffer.allocation.PinnedMemoryStrategy;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.impl.AccumulationKernelCall;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.impl.IndexAccumulationKernelCall;


import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;


/**
 * Wraps the generation of kernel parameters
 * , creating, copying
 * and destroying any cuda device allocations
 *
 * Allocator Pass Note 17/02/2016:
 * We don't destroy anything here anymore, since that mechanics was moved to new Allocator mechanics.
 * However, we still can use AutoCloseable as a nice place for syncStream hook  :)
 *
 * @author bam4d
 * @author raver119@gmail.com
 *
 */
public class KernelParamsWrapper implements AutoCloseable {


    private boolean closeContext;

    private CudaContext context;

    private boolean scalarResult;

    /**
     * List of processed kernel parameters ready to be passed to the kernel
     */
    final public Object[] kernelParameters;

    /**
     * The pointers that need to be freed as part of this closable resource
     */
    //final List<CublasPointer> pointersToFree;

    /**
     * The pointers that have results that need to be passed back to host buffers
     */
    final List<CublasPointer> resultPointers;

    /**
     * The operation that should receive the result
     */
    private Op resultOp;

    /**
     * The list of processed kernel parameters, These should be get passed to the cuda kernel
     * @return
     */
    public Object[] getKernelParameters() {
        return kernelParameters;
    }

    /**
     * conversion list of arrays to their assigned cublas pointer
     */
    private Multimap<INDArray, CublasPointer> arrayToPointer;

    private int resultLength = 1;


    /**
     * set the array that will contain the results, If the array is not set, then data from the device will not be copied to the host
     * @param array
     * @return
     */
    public KernelParamsWrapper setResultArray(INDArray array) {
        if(!arrayToPointer.containsKey(array)) {
          throw new IllegalStateException("No array found: unable to set array value");
        }
        CublasPointer resultPointer = arrayToPointer.get(array).iterator().next();
        resultPointer.setResultPointer(true);
        if(resultPointer == null) {
            throw new RuntimeException("Results array must be supplied as a kernel parameter");
        }

        resultPointers.add(resultPointer);

        return this;
    }
    /**
     * set the Op that this result is for
     * @param op
     * @param result
     * @return
     */
    public KernelParamsWrapper setResultOp(IndexAccumulation op, INDArray result,int...dimension) {
        resultOp = op;
        resultLength = result.length();
        scalarResult = (dimension == null || dimension.length < 1 || dimension[0] == Integer.MAX_VALUE || result.length() == 1);
        setResultArray(result);
        return this;
    }

    /**
     * set the Op that this result is for
     * @param op
     * @param result
     * @return
     */
    public KernelParamsWrapper setResultOp(Accumulation op, INDArray result,int...dimension) {
        resultOp = op;
        resultLength = result.length();
        scalarResult = (dimension == null || dimension.length < 1 || dimension[0] == Integer.MAX_VALUE || result.length() == 1);
        setResultArray(result);
        return this;
    }
    /**
     * Create a new wrapper for the kernel parameters.
     *
     * This wrapper manages the host - and device communication and.
     *
     * To set the result on a specific operation, use setResultOp()
     * To set the array which is the result INDArray, use setResultArray()
     * @param kernelParams
     */
    public KernelParamsWrapper(Object... kernelParams) {
        this(false, kernelParams);
    }
    /**
     * Create a new wrapper for the kernel parameters.
     *
     * This wrapper manages the host - and device communication and.
     *
     * To set the result on a specific operation, use setResultOp()
     * To set the array which is the result INDArray, use setResultArray()
     * @param kernelParams
     */
    public KernelParamsWrapper(boolean closeContext,Object... kernelParams) {
        resultPointers = new ArrayList<>();
        context = new CudaContext(closeContext);

        CudaArgs.ArgsAndReferences argsAndReferences = CudaArgs.argsAndReference(context,kernelParams);
        kernelParameters = argsAndReferences.getArgs();

        //arrayToPointer = argsAndReferences.getArrayToPointer();
        //pointersToFree = argsAndReferences.getPointersToFree();

        // This is not used anymore, since we're using external stream management
//        context.initOldStream();
//        context.initStream();
  //      this.closeContext = closeContext;
    }

    /**
     * Free all the buffers from this kernel's parameters
     */
    @Override
    public void close() throws Exception {
        if(context.getOldStream() != null)
            context.syncOldStream();
        if(context.getStream() != null)
            context.syncStream();


        /*
        for(CublasPointer cublasPointer : pointersToFree) {
            if(resultPointers.contains(cublasPointer)) {
                //sets the result for the buffer
                //since this ends up being a scalar
                if(closeContext) {
                    if(scalarResult && resultOp instanceof Accumulation || resultOp instanceof IndexAccumulation) {
                        setResultForOp(resultOp, cublasPointer);
                    }
                    else
                        cublasPointer.copyToHost();
                    cublasPointer.close();
                }
                else
                    context.setResultPointer(cublasPointer);
            }

        }

        if(closeContext)
            context.destroy();
        */
    }

    /**
     * Set the result within the accumulation operation
     * @param acc
     * @param devicePointer
     */
    @Deprecated
    private void setResultForOp(Op acc, CublasPointer devicePointer) {
        if (devicePointer.getBuffer().dataType() == DataBuffer.Type.DOUBLE) {
            if(ContextHolder.getInstance().getMemoryStrategy() instanceof PinnedMemoryStrategy) {
                ByteBuffer buff = devicePointer.getHostPointer().getByteBuffer(0,acc.x().data().getElementSize() * resultLength);
                buff.order(ByteOrder.nativeOrder());
                INDArray setResult = Nd4j.create(Nd4j.createBuffer(buff, DataBuffer.Type.DOUBLE,resultLength));
                int oldN = acc.n();
                acc.setX(setResult);
                acc.setN(oldN);
                if(acc instanceof IndexAccumulation && resultLength > 1)
                    IndexAccumulationKernelCall.calculateBlockResult((IndexAccumulation) acc, setResult);

                else if(acc instanceof Accumulation && resultLength > 1)
                    AccumulationKernelCall.calculateBlockResult((Accumulation) acc, setResult);
                else if(acc instanceof Accumulation) {
                    Accumulation acc2 = (Accumulation) acc;
                    acc2.setFinalResult(setResult.getDouble(0));
                }
                else if(acc instanceof IndexAccumulation) {
                    IndexAccumulation acc2 = (IndexAccumulation) acc;
                    acc2.setFinalResult(setResult.getInt(0));
                }
            }
            else {
                double[] data = new double[resultLength];
                Pointer get = Pointer.to(data);
                JCuda.cudaMemcpyAsync(
                        get
                        , devicePointer.getDevicePointer()
                        , resultLength * Sizeof.DOUBLE
                        , cudaMemcpyKind.cudaMemcpyDeviceToHost
                        , context.getOldStream());
                context.syncOldStream();

            }


        }
        else if (devicePointer.getBuffer().dataType() == DataBuffer.Type.FLOAT) {
            if(ContextHolder.getInstance().getMemoryStrategy() instanceof PinnedMemoryStrategy) {
                ByteBuffer buff = devicePointer.getHostPointer().getByteBuffer(0,acc.x().data().getElementSize() * resultLength);
                buff.order(ByteOrder.nativeOrder());
                INDArray setResult = Nd4j.create(Nd4j.createBuffer(buff, DataBuffer.Type.FLOAT,resultLength));
                if(acc instanceof IndexAccumulation)
                    IndexAccumulationKernelCall.calculateBlockResult((IndexAccumulation) acc,setResult);
                else if(acc instanceof Accumulation)
                    AccumulationKernelCall.calculateBlockResult((Accumulation) acc,setResult);
            }
            else {
                float[] data = new float[resultLength];
                Pointer get = Pointer.to(data);
                JCuda.cudaMemcpyAsync(
                        get
                        , devicePointer.getDevicePointer()
                        , resultLength * Sizeof.FLOAT
                        , cudaMemcpyKind.cudaMemcpyDeviceToHost
                        , context.getOldStream());
                context.syncOldStream();

            }
        }
    }

    public CudaContext getContext() {
        return context;
    }




}