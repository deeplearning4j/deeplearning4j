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
package org.deeplearning4j.nn.layers.convolution.subsampling;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.ShortPointer;
import org.bytedeco.javacpp.indexer.HalfIndexer;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;

import static org.bytedeco.javacpp.cuda.CUstream_st;
import static org.bytedeco.javacpp.cuda.cudaSuccess;
import static org.bytedeco.javacpp.cudnn.*;

/**
 * cuDNN-based helper for the subsampling layer.
 *
 * @author saudet
 */
public class CudnnSubsamplingHelper implements SubsamplingHelper {

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
        cudnnPoolingStruct poolingDesc = new cudnnPoolingStruct();

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
            poolingDesc = new cudnnPoolingStruct(c.poolingDesc);
        }

        void createHandles() {
            checkCudnn(cudnnCreate(this));
            checkCudnn(cudnnCreateTensorDescriptor(srcTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(dstTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(deltaTensorDesc));
            checkCudnn(cudnnCreatePoolingDescriptor(poolingDesc));
        }

        void destroyHandles() {
            checkCudnn(cudnnDestroyPoolingDescriptor(poolingDesc));
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
    INDArray reduced = null;

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray epsilon,
            int[] kernel, int[] strides, int[] pad, PoolingType poolingType) {
        int miniBatch = input.size(0);
        int depth = input.size(1);
        int inH = input.size(2);
        int inW = input.size(3);

        int outH = Convolution.outSize(inH, kernel[0], strides[0], pad[0],false);
        int outW = Convolution.outSize(inW, kernel[1], strides[1], pad[1], false);

        //subsampling doesn't have weights and thus gradients are not calculated for this layer
        //only scale and reshape epsilon
        Gradient retGradient = new DefaultGradient();

        //Epsilons in shape: [miniBatch, depth, outH, outW]
        //Epsilons out shape: [miniBatch, depth, inH, inW]


        int poolingMode;
        switch(poolingType) {
            case AVG:
                poolingMode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
                break;
            case MAX:
                poolingMode = CUDNN_POOLING_MAX;
                break;
            case NONE:
                return new Pair<>(retGradient, epsilon);
            default:
                return null;
        }

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
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.deltaTensorDesc, dataType, miniBatch, depth, outH, outW,
                deltaStride[0], deltaStride[1], deltaStride[2], deltaStride[3]));
        checkCudnn(cudnnSetPooling2dDescriptor(cudnnContext.poolingDesc, poolingMode, CUDNN_PROPAGATE_NAN,
                kernel[0], kernel[1], pad[0], pad[1], strides[0], strides[1]));

        INDArray outEpsilon = Nd4j.create(new int[]{miniBatch,depth,inH,inW},'c');
        int[] dstStride = outEpsilon.stride();
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.dstTensorDesc, dataType, miniBatch, depth, inH, inW,
                dstStride[0], dstStride[1], dstStride[2], dstStride[3]));

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareAction(input, epsilon, reduced, outEpsilon);
        Pointer srcData = allocator.getPointer(input, context);
        Pointer epsData = allocator.getPointer(epsilon, context);
        Pointer zData = allocator.getPointer(reduced, context);
        Pointer dstData = allocator.getPointer(outEpsilon, context);

        checkCudnn(cudnnSetStream(cudnnContext, new CUstream_st(context.getOldStream())));
        checkCudnn(cudnnPoolingBackward(cudnnContext, cudnnContext.poolingDesc, alpha, cudnnContext.deltaTensorDesc, zData,
                cudnnContext.deltaTensorDesc, epsData, cudnnContext.srcTensorDesc, srcData, beta, cudnnContext.dstTensorDesc, dstData));

        allocator.registerAction(context, input, epsilon, reduced, outEpsilon);

        return new Pair<>(retGradient,outEpsilon);
    }


    @Override
    public INDArray activate(INDArray input, boolean training,
            int[] kernel, int[] strides, int[] pad, PoolingType poolingType) {
        int miniBatch = input.size(0);
        int inDepth = input.size(1);
        int inH = input.size(2);
        int inW = input.size(3);

        int outH = Convolution.outSize(inH, kernel[0], strides[0], pad[0],false);
        int outW = Convolution.outSize(inW, kernel[1], strides[1], pad[1], false);

        int poolingMode;
        switch(poolingType) {
            case AVG:
                poolingMode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
                break;
            case MAX:
                poolingMode = CUDNN_POOLING_MAX;
                break;
            case NONE:
                return input;
            default:
                return null;
        }

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner)Nd4j.getExecutioner()).flushQueue();

        int[] srcStride = input.stride();
        checkCudnn(cudnnSetPooling2dDescriptor(cudnnContext.poolingDesc, poolingMode, CUDNN_PROPAGATE_NAN,
                kernel[0], kernel[1], pad[0], pad[1], strides[0], strides[1]));
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.srcTensorDesc, dataType, miniBatch, inDepth, inH, inW,
                srcStride[0], srcStride[1], srcStride[2], srcStride[3]));

        reduced = Nd4j.createUninitialized(new int[]{miniBatch,inDepth,outH,outW},'c');
        int[] dstStride = reduced.stride();
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.dstTensorDesc, dataType, miniBatch, inDepth, outH, outW,
                dstStride[0], dstStride[1], dstStride[2], dstStride[3]));

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareAction(input, reduced);
        Pointer srcData = allocator.getPointer(input, context);
        Pointer dstData = allocator.getPointer(reduced, context);

        checkCudnn(cudnnSetStream(cudnnContext, new CUstream_st(context.getOldStream())));
        checkCudnn(cudnnPoolingForward(cudnnContext, cudnnContext.poolingDesc,
                alpha, cudnnContext.srcTensorDesc, srcData, beta, cudnnContext.dstTensorDesc, dstData));

        allocator.registerAction(context, input, reduced);

        return reduced;
    }

}
