/*
 *
 *  * Copyright 2016 Skymind,Inc.
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
 */
package org.deeplearning4j.nn.layers.normalization;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.ShortPointer;
import org.bytedeco.javacpp.indexer.HalfIndexer;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;

import static org.bytedeco.javacpp.cuda.CUstream_st;
import static org.bytedeco.javacpp.cuda.cudaSuccess;
import static org.bytedeco.javacpp.cudnn.*;

/**
 * cuDNN-based helper for the local response normalization layer.
 *
 * @author saudet
 */
public class CudnnLocalResponseNormalizationHelper implements LocalResponseNormalizationHelper {

    static void checkCuda(int error) {
        if (error != cudaSuccess) {
            throw new RuntimeException("CUDA error = " + error);
        }
    }

    static void checkCudnn(int status) {
        if (status != CUDNN_STATUS_SUCCESS) {
            throw new RuntimeException("cuDNN status = " + status);
        }
    }

    static class CudnnContext extends cudnnContext {

        static class Deallocator extends CudnnContext implements Pointer.Deallocator {
            Deallocator(CudnnContext c) { super(c); }
            @Override public void deallocate() { destroyHandles(); }
        }

        cudnnTensorStruct srcTensorDesc = new cudnnTensorStruct(),
                          dstTensorDesc = new cudnnTensorStruct(),
                          deltaTensorDesc = new cudnnTensorStruct();
        cudnnLRNStruct lrnDesc = new cudnnLRNStruct();

        CudnnContext() {
            // insure that cuDNN initializes on the same device as ND4J for this thread
            Nd4j.create(1);
            createHandles();
            deallocator(new Deallocator(this));
        }

        CudnnContext(CudnnContext c) {
            super(c);
            srcTensorDesc = new cudnnTensorStruct(c.srcTensorDesc);
            dstTensorDesc = new cudnnTensorStruct(c.dstTensorDesc);
            deltaTensorDesc = new cudnnTensorStruct(c.deltaTensorDesc);
            lrnDesc = new cudnnLRNStruct(c.lrnDesc);
        }

        void createHandles() {
            checkCudnn(cudnnCreate(this));
            checkCudnn(cudnnCreateTensorDescriptor(srcTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(dstTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(deltaTensorDesc));
            checkCudnn(cudnnCreateLRNDescriptor(lrnDesc));
        }

        void destroyHandles() {
            checkCudnn(cudnnDestroyLRNDescriptor(lrnDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(srcTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(dstTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(deltaTensorDesc));
            checkCudnn(cudnnDestroy(this));
        }
    }

    CudnnContext cudnnContext = new CudnnContext();
    int dataType = Nd4j.dataType() == DataBuffer.Type.DOUBLE ? CUDNN_DATA_DOUBLE : Nd4j.dataType() == DataBuffer.Type.FLOAT ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF;
    int tensorFormat = CUDNN_TENSOR_NCHW;
    Pointer alpha = Nd4j.dataType() == DataBuffer.Type.DOUBLE ? new DoublePointer(1.0)
                  : Nd4j.dataType() == DataBuffer.Type.FLOAT ? new FloatPointer(1.0f)
                  : new ShortPointer(new short[] {(short)HalfIndexer.fromFloat(1.0f)});
    Pointer beta  = Nd4j.dataType() == DataBuffer.Type.DOUBLE ? new DoublePointer(0.0)
                  : Nd4j.dataType() == DataBuffer.Type.FLOAT ? new FloatPointer(0.0f)
                  : new ShortPointer(new short[] {(short)HalfIndexer.fromFloat(0.0f)});;
    INDArray activations = null;

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray epsilon, double k, double n, double alpha, double beta) {
        int miniBatch = input.size(0);
        int depth = input.size(1);
        int inH = input.size(2);
        int inW = input.size(3);

        Gradient retGradient = new DefaultGradient();

        if (!Shape.strideDescendingCAscendingF(epsilon)) {
            // apparently not supported by cuDNN
            epsilon = epsilon.dup();
        }

        int[] srcStride = input.stride();
        int[] deltaStride = epsilon.stride();

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner)Nd4j.getExecutioner()).flushQueue();

        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.srcTensorDesc, dataType, miniBatch, depth, inH, inW,
                srcStride[0], srcStride[1], srcStride[2], srcStride[3]));
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.deltaTensorDesc, dataType, miniBatch, depth, inH, inW,
                deltaStride[0], deltaStride[1], deltaStride[2], deltaStride[3]));
        checkCudnn(cudnnSetLRNDescriptor(cudnnContext.lrnDesc, (int)n, alpha, beta, k));

        INDArray nextEpsilon = Nd4j.createUninitialized(new int[]{miniBatch,depth,inH,inW},'c');
        int[] dstStride = nextEpsilon.stride();
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.dstTensorDesc, dataType, miniBatch, depth, inH, inW,
                dstStride[0], dstStride[1], dstStride[2], dstStride[3]));

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareActionAllWrite(input, epsilon, activations, nextEpsilon);
        Pointer srcData = allocator.getPointer(input, context);
        Pointer epsData = allocator.getPointer(epsilon, context);
        Pointer zData = allocator.getPointer(activations, context);
        Pointer dstData = allocator.getPointer(nextEpsilon, context);

        checkCudnn(cudnnSetStream(cudnnContext, new CUstream_st(context.getOldStream())));
        checkCudnn(cudnnLRNCrossChannelBackward(cudnnContext, cudnnContext.lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
                this.alpha, cudnnContext.deltaTensorDesc, zData, cudnnContext.deltaTensorDesc, epsData, cudnnContext.srcTensorDesc, srcData,
                this.beta, cudnnContext.dstTensorDesc, dstData));

        allocator.getFlowController().registerActionAllWrite(context, input, epsilon, activations, nextEpsilon);

        return new Pair<>(retGradient,nextEpsilon);
    }


    @Override
    public INDArray activate(INDArray input, boolean training, double k, double n, double alpha, double beta) {
        int miniBatch = input.size(0);
        int inDepth = input.size(1);
        int inH = input.size(2);
        int inW = input.size(3);

        int[] srcStride = input.stride();
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.srcTensorDesc, dataType, miniBatch, inDepth, inH, inW,
                srcStride[0], srcStride[1], srcStride[2], srcStride[3]));

        activations = Nd4j.createUninitialized(new int[]{miniBatch,inDepth,inH,inW},'c');
        int[] dstStride = activations.stride();
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.dstTensorDesc, dataType, miniBatch, inDepth, inH, inW,
                dstStride[0], dstStride[1], dstStride[2], dstStride[3]));
        checkCudnn(cudnnSetLRNDescriptor(cudnnContext.lrnDesc, (int)n, alpha, beta, k));

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareActionAllWrite(input, activations);
        Pointer srcData = allocator.getPointer(input, context);
        Pointer dstData = allocator.getPointer(activations, context);

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner)Nd4j.getExecutioner()).flushQueue();

        checkCudnn(cudnnSetStream(cudnnContext, new CUstream_st(context.getOldStream())));
        checkCudnn(cudnnLRNCrossChannelForward(cudnnContext, cudnnContext.lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
                this.alpha, cudnnContext.srcTensorDesc, srcData, this.beta, cudnnContext.dstTensorDesc, dstData));

        allocator.getFlowController().registerActionAllWrite(context, input, activations);

        return activations;
    }

}
