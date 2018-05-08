/*-
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

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseCudnnHelper;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import static org.bytedeco.javacpp.cuda.CUstream_st;
import static org.bytedeco.javacpp.cudnn.*;

/**
 * cuDNN-based helper for the local response normalization layer.
 *
 * @author saudet
 */
@Slf4j
public class CudnnLocalResponseNormalizationHelper extends BaseCudnnHelper implements LocalResponseNormalizationHelper {

    private static class CudnnLocalResponseNormalizationContext extends CudnnContext {

        private static class Deallocator extends CudnnLocalResponseNormalizationContext implements Pointer.Deallocator {
            Deallocator(CudnnLocalResponseNormalizationContext c) {
                super(c);
            }

            @Override
            public void deallocate() {
                destroyHandles();
            }
        }

        private cudnnTensorStruct srcTensorDesc = new cudnnTensorStruct(), dstTensorDesc = new cudnnTensorStruct(),
                        deltaTensorDesc = new cudnnTensorStruct();
        private cudnnLRNStruct lrnDesc = new cudnnLRNStruct();

        public CudnnLocalResponseNormalizationContext() {
            createHandles();
            deallocator(new Deallocator(this));
        }

        public CudnnLocalResponseNormalizationContext(CudnnLocalResponseNormalizationContext c) {
            super(c);
            srcTensorDesc = new cudnnTensorStruct(c.srcTensorDesc);
            dstTensorDesc = new cudnnTensorStruct(c.dstTensorDesc);
            deltaTensorDesc = new cudnnTensorStruct(c.deltaTensorDesc);
            lrnDesc = new cudnnLRNStruct(c.lrnDesc);
        }

        @Override
        protected void createHandles() {
            super.createHandles();
            checkCudnn(cudnnCreateTensorDescriptor(srcTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(dstTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(deltaTensorDesc));
            checkCudnn(cudnnCreateLRNDescriptor(lrnDesc));
        }

        @Override
        protected void destroyHandles() {
            checkCudnn(cudnnDestroyLRNDescriptor(lrnDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(srcTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(dstTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(deltaTensorDesc));
            super.destroyHandles();
        }
    }

    private CudnnLocalResponseNormalizationContext cudnnContext = new CudnnLocalResponseNormalizationContext();
    private INDArray activations = null;

    public boolean checkSupported(double k, double n, double alpha, double beta) {
        boolean supported = checkSupported();
        if (n < CUDNN_LRN_MIN_N) {
            supported = false;
            log.warn("Not supported: n < CUDNN_LRN_MIN_N (" + n + " < " + CUDNN_LRN_MIN_N + ")");
        }
        if (n > CUDNN_LRN_MAX_N) {
            supported = false;
            log.warn("Not supported: n > CUDNN_LRN_MAX_N (" + n + " > " + CUDNN_LRN_MAX_N + ")");
        }
        if (k < CUDNN_LRN_MIN_K) {
            supported = false;
            log.warn("Not supported: k < CUDNN_LRN_MIN_K (" + k + " < " + CUDNN_LRN_MIN_K + ")");
        }
        if (beta < CUDNN_LRN_MIN_BETA) {
            supported = false;
            log.warn("Not supported: beta < CUDNN_LRN_MIN_BETA (" + beta + " < " + CUDNN_LRN_MIN_BETA + ")");
        }
        return supported;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray epsilon, double k, double n, double alpha,
                                                     double beta, LayerWorkspaceMgr workspaceMgr) {
        int miniBatch = input.size(0);
        int depth = input.size(1);
        int inH = input.size(2);
        int inW = input.size(3);

        Gradient retGradient = new DefaultGradient();

        if (!Shape.hasDefaultStridesForShape(epsilon)) {
            // apparently not supported by cuDNN
            epsilon = epsilon.dup('c');
        }

        int[] srcStride = input.stride();
        int[] deltaStride = epsilon.stride();

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.srcTensorDesc, dataType, miniBatch, depth, inH, inW,
                        srcStride[0], srcStride[1], srcStride[2], srcStride[3]));
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.deltaTensorDesc, dataType, miniBatch, depth, inH, inW,
                        deltaStride[0], deltaStride[1], deltaStride[2], deltaStride[3]));
        checkCudnn(cudnnSetLRNDescriptor(cudnnContext.lrnDesc, (int) n, alpha, beta, k));

        INDArray nextEpsilon = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, new int[] {miniBatch, depth, inH, inW}, 'c');
        int[] dstStride = nextEpsilon.stride();
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.dstTensorDesc, dataType, miniBatch, depth, inH, inW,
                        dstStride[0], dstStride[1], dstStride[2], dstStride[3]));

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context =
                        allocator.getFlowController().prepareActionAllWrite(input, epsilon, activations, nextEpsilon);
        Pointer srcData = allocator.getPointer(input, context);
        Pointer epsData = allocator.getPointer(epsilon, context);
        Pointer zData = allocator.getPointer(activations, context);
        Pointer dstData = allocator.getPointer(nextEpsilon, context);

        checkCudnn(cudnnSetStream(cudnnContext, new CUstream_st(context.getOldStream())));
        checkCudnn(cudnnLRNCrossChannelBackward(cudnnContext, cudnnContext.lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
                        this.alpha, cudnnContext.deltaTensorDesc, zData, cudnnContext.deltaTensorDesc, epsData,
                        cudnnContext.srcTensorDesc, srcData, this.beta, cudnnContext.dstTensorDesc, dstData));

        allocator.getFlowController().registerActionAllWrite(context, input, epsilon, activations, nextEpsilon);

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            context.syncOldStream();

        return new Pair<>(retGradient, nextEpsilon);
    }


    @Override
    public INDArray activate(INDArray input, boolean training, double k, double n, double alpha, double beta, LayerWorkspaceMgr workspaceMgr) {
        int miniBatch = input.size(0);
        int inDepth = input.size(1);
        int inH = input.size(2);
        int inW = input.size(3);

        if(!Shape.hasDefaultStridesForShape(input)){
            input = input.dup('c');
        }

        int[] srcStride = input.stride();
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.srcTensorDesc, dataType, miniBatch, inDepth, inH, inW,
                        srcStride[0], srcStride[1], srcStride[2], srcStride[3]));

        activations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new int[] {miniBatch, inDepth, inH, inW}, 'c');
        int[] dstStride = activations.stride();
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.dstTensorDesc, dataType, miniBatch, inDepth, inH, inW,
                        dstStride[0], dstStride[1], dstStride[2], dstStride[3]));
        checkCudnn(cudnnSetLRNDescriptor(cudnnContext.lrnDesc, (int) n, alpha, beta, k));

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareActionAllWrite(input, activations);
        Pointer srcData = allocator.getPointer(input, context);
        Pointer dstData = allocator.getPointer(activations, context);

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        checkCudnn(cudnnSetStream(cudnnContext, new CUstream_st(context.getOldStream())));
        checkCudnn(cudnnLRNCrossChannelForward(cudnnContext, cudnnContext.lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
                        this.alpha, cudnnContext.srcTensorDesc, srcData, this.beta, cudnnContext.dstTensorDesc,
                        dstData));

        allocator.getFlowController().registerActionAllWrite(context, input, activations);

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            context.syncOldStream();

        return activations;
    }

}
