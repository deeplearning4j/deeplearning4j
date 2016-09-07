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
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bytedeco.javacpp.cuda.*;
import static org.bytedeco.javacpp.cudnn.*;

/**
 * cuDNN-based helper for the batch normalization layer.
 *
 * @author saudet
 */
public class CudnnBatchNormalizationHelper implements BatchNormalizationHelper {
    protected static final Logger log = LoggerFactory.getLogger(CudnnBatchNormalizationHelper.class);

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
                          deltaTensorDesc = new cudnnTensorStruct(),
                          gammaBetaTensorDesc = new cudnnTensorStruct();

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
            gammaBetaTensorDesc = new cudnnTensorStruct(c.gammaBetaTensorDesc);
        }

        void createHandles() {
            checkCudnn(cudnnCreate(this));
            checkCudnn(cudnnCreateTensorDescriptor(srcTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(dstTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(deltaTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(gammaBetaTensorDesc));
        }

        void destroyHandles() {
            checkCudnn(cudnnDestroyTensorDescriptor(srcTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(dstTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(deltaTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(gammaBetaTensorDesc));
            checkCudnn(cudnnDestroy(this));
        }
    }

    static class Cache extends Pointer {

        static class Deallocator extends Cache implements Pointer.Deallocator {
            Deallocator(Cache c) { super(c); }
            @Override public void deallocate() { checkCuda(cudaFree(this)); setNull(); }
        }

        static class HostDeallocator extends Cache implements Pointer.Deallocator {
            HostDeallocator(Cache c) { super(c); }
            @Override public void deallocate() { checkCuda(cudaFreeHost(this)); setNull(); }
        }

        Cache() { }

        Cache(long size) {
            position = 0;
            limit = capacity = size;
            int error = cudaMalloc(this, size);
            if (error != cudaSuccess) {
                log.warn("Cannot allocate " + size + " bytes of device memory (CUDA error = " + error + "), proceeding with host memory");
                checkCuda(cudaMallocHost(this, size));
                deallocator(new HostDeallocator(this));
            } else {
                deallocator(new Deallocator(this));
            }
        }

        Cache(Cache c) {
            super(c);
        }
    }

    CudnnContext cudnnContext = new CudnnContext();
    Cache meanCache = new Cache();
    Cache varCache = new Cache();
    int dataType = Nd4j.dataType() == DataBuffer.Type.DOUBLE ? CUDNN_DATA_DOUBLE : Nd4j.dataType() == DataBuffer.Type.FLOAT ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF;
    int tensorFormat = CUDNN_TENSOR_NCHW;
    int batchNormMode = CUDNN_BATCHNORM_SPATIAL; // would need to increase rank of gamma and beta for CUDNN_BATCHNORM_PER_ACTIVATION
    Pointer alpha = Nd4j.dataType() == DataBuffer.Type.DOUBLE ? new DoublePointer(1.0)
                  : Nd4j.dataType() == DataBuffer.Type.FLOAT ? new FloatPointer(1.0f)
                  : new ShortPointer(new short[] {(short)HalfIndexer.fromFloat(1.0f)});
    Pointer beta  = Nd4j.dataType() == DataBuffer.Type.DOUBLE ? new DoublePointer(0.0)
                  : Nd4j.dataType() == DataBuffer.Type.FLOAT ? new FloatPointer(0.0f)
                  : new ShortPointer(new short[] {(short)HalfIndexer.fromFloat(0.0f)});;

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray epsilon,
            int[] shape, INDArray gamma, INDArray dGammaView, INDArray dBetaView, double eps) {
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

        INDArray nextEpsilon = Nd4j.createUninitialized(new int[]{miniBatch,depth,inH,inW},'c');
        int[] dstStride = nextEpsilon.stride();
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.dstTensorDesc, dataType, miniBatch, depth, inH, inW,
                dstStride[0], dstStride[1], dstStride[2], dstStride[3]));
        int[] gammaStride = gamma.stride();
        checkCudnn(cudnnSetTensor4dDescriptor(cudnnContext.gammaBetaTensorDesc, tensorFormat, dataType,
                shape[0], shape[1], shape.length > 2 ? shape[2] : 1, shape.length > 3 ? shape[3] : 1));

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareAction(input, epsilon, nextEpsilon, gamma, dGammaView, dBetaView);
        Pointer srcData = allocator.getPointer(input, context);
        Pointer epsData = allocator.getPointer(epsilon, context);
        Pointer dstData = allocator.getPointer(nextEpsilon, context);
        Pointer gammaData = allocator.getPointer(gamma, context);
        Pointer dGammaData = allocator.getPointer(dGammaView, context);
        Pointer dBetaData = allocator.getPointer(dBetaView, context);

        checkCudnn(cudnnSetStream(cudnnContext, new CUstream_st(context.getOldStream())));
        checkCudnn(cudnnBatchNormalizationBackward(cudnnContext, batchNormMode, alpha, beta, alpha, alpha,
                cudnnContext.srcTensorDesc, srcData, cudnnContext.deltaTensorDesc, epsData, cudnnContext.dstTensorDesc, dstData,
                cudnnContext.gammaBetaTensorDesc, gammaData, dGammaData, dBetaData, eps, meanCache, varCache));

        allocator.registerAction(context, input, epsilon, nextEpsilon, gamma, dGammaView, dBetaView);

        retGradient.setGradientFor(BatchNormalizationParamInitializer.GAMMA, dGammaView);
        retGradient.setGradientFor(BatchNormalizationParamInitializer.BETA, dBetaView);
        return new Pair<>(retGradient,nextEpsilon);
    }


    @Override
    public INDArray preOutput(INDArray x, boolean training, int[] shape,
            INDArray gamma, INDArray beta, INDArray mean, INDArray var, double decay, double eps) {
        int miniBatch = x.size(0);
        int inDepth = x.size(1);
        int inH = x.size(2);
        int inW = x.size(3);

        int[] srcStride = x.stride();
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.srcTensorDesc, dataType, miniBatch, inDepth, inH, inW,
                srcStride[0], srcStride[1], srcStride[2], srcStride[3]));

        INDArray activations = Nd4j.createUninitialized(new int[]{miniBatch,inDepth,inH,inW},'c');
        int[] dstStride = activations.stride();
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.dstTensorDesc, dataType, miniBatch, inDepth, inH, inW,
                dstStride[0], dstStride[1], dstStride[2], dstStride[3]));
        int[] gammaStride = gamma.stride();
        checkCudnn(cudnnSetTensor4dDescriptor(cudnnContext.gammaBetaTensorDesc, tensorFormat, dataType,
                shape[0], shape[1], shape.length > 2 ? shape[2] : 1, shape.length > 3 ? shape[3] : 1));

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareAction(x, activations, gamma, beta, mean, var);
        Pointer srcData = allocator.getPointer(x, context);
        Pointer dstData = allocator.getPointer(activations, context);
        Pointer gammaData = allocator.getPointer(gamma, context);
        Pointer betaData = allocator.getPointer(beta, context);
        Pointer meanData = allocator.getPointer(mean, context);
        Pointer varData = allocator.getPointer(var, context);

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner)Nd4j.getExecutioner()).flushQueue();

        checkCudnn(cudnnSetStream(cudnnContext, new CUstream_st(context.getOldStream())));
        if (training) {
            if (meanCache.capacity() < mean.data().length() * mean.data().getElementSize()) {
                meanCache.deallocate();
                meanCache = new Cache(mean.data().length() * mean.data().getElementSize());
            }
            if (varCache.capacity() < var.data().length() * mean.data().getElementSize()) {
                varCache.deallocate();
                varCache = new Cache(var.data().length() * mean.data().getElementSize());
            }
            checkCudnn(cudnnBatchNormalizationForwardTraining(cudnnContext, batchNormMode, this.alpha, this.beta,
                    cudnnContext.srcTensorDesc, srcData, cudnnContext.dstTensorDesc, dstData,
                    cudnnContext.gammaBetaTensorDesc, gammaData, betaData, decay, meanData, varData, eps, meanCache, varCache));
        } else {
            checkCudnn(cudnnBatchNormalizationForwardInference(cudnnContext, batchNormMode, this.alpha, this.beta,
                    cudnnContext.srcTensorDesc, srcData, cudnnContext.dstTensorDesc, dstData,
                    cudnnContext.gammaBetaTensorDesc, gammaData, betaData, meanData, varData, eps));
        }

        allocator.registerAction(context, x, activations, gamma, beta, mean, var);

        return activations;
    }

}
