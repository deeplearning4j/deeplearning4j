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
package org.deeplearning4j.nn.layers.convolution;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.*;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.BwdFilterAlgo;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.BwdDataAlgo;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.FwdAlgo;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.BaseCudnnHelper;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;

import static org.bytedeco.javacpp.cuda.*;
import static org.bytedeco.javacpp.cudnn.*;

/**
 * cuDNN-based helper for the convolution layer.
 *
 * @author saudet
 */
@Slf4j
public class CudnnConvolutionHelper extends BaseCudnnHelper implements ConvolutionHelper {

    private static class CudnnConvolutionContext extends CudnnContext {

        private static class Deallocator extends CudnnConvolutionContext implements Pointer.Deallocator {
            Deallocator(CudnnConvolutionContext c) {
                super(c);
            }

            @Override
            public void deallocate() {
                destroyHandles();
            }
        }

        private cudnnTensorStruct srcTensorDesc = new cudnnTensorStruct(), dstTensorDesc = new cudnnTensorStruct(),
                        biasTensorDesc = new cudnnTensorStruct(), deltaTensorDesc = new cudnnTensorStruct();
        private cudnnFilterStruct filterDesc = new cudnnFilterStruct();
        private cudnnConvolutionStruct convDesc = new cudnnConvolutionStruct();
        private cudnnActivationStruct activationDesc = new cudnnActivationStruct();

        public CudnnConvolutionContext() {
            createHandles();
            deallocator(new Deallocator(this));
        }

        public CudnnConvolutionContext(CudnnConvolutionContext c) {
            super(c);
            srcTensorDesc = new cudnnTensorStruct(c.srcTensorDesc);
            dstTensorDesc = new cudnnTensorStruct(c.dstTensorDesc);
            biasTensorDesc = new cudnnTensorStruct(c.biasTensorDesc);
            deltaTensorDesc = new cudnnTensorStruct(c.deltaTensorDesc);
            filterDesc = new cudnnFilterStruct(c.filterDesc);
            convDesc = new cudnnConvolutionStruct(c.convDesc);
            activationDesc = new cudnnActivationStruct(c.activationDesc);
        }

        @Override
        protected void createHandles() {
            super.createHandles();
            checkCudnn(cudnnCreateTensorDescriptor(srcTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(dstTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(biasTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(deltaTensorDesc));
            checkCudnn(cudnnCreateFilterDescriptor(filterDesc));
            checkCudnn(cudnnCreateConvolutionDescriptor(convDesc));
            checkCudnn(cudnnCreateActivationDescriptor(activationDesc));
        }

        @Override
        protected void destroyHandles() {
            checkCudnn(cudnnDestroyActivationDescriptor(activationDesc));
            checkCudnn(cudnnDestroyConvolutionDescriptor(convDesc));
            checkCudnn(cudnnDestroyFilterDescriptor(filterDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(srcTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(dstTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(biasTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(deltaTensorDesc));
            super.destroyHandles();
        }
    }

    private CudnnConvolutionContext cudnnContext = new CudnnConvolutionContext();
    private DataCache workSpace = new DataCache();

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray weights, INDArray delta, int[] kernel,
                    int[] strides, int[] pad, INDArray biasGradView, INDArray weightGradView, IActivation afn,
                    AlgoMode mode, BwdFilterAlgo bwdFilterAlgo, BwdDataAlgo bwdDataAlgo,
                    ConvolutionMode convolutionMode) {
        int miniBatch = input.size(0);
        int inH = input.size(2);
        int inW = input.size(3);

        int outDepth = weights.size(0);
        int inDepth = weights.size(1);
        int kH = weights.size(2);
        int kW = weights.size(3);

        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, null, convolutionMode); //Also performs validation
            pad = ConvolutionUtils.getSameModeBottomRightPadding(outSize, new int[] {inH, inW}, kernel, strides);
        } else {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, pad, convolutionMode); //Also performs validation
        }

        int outH = outSize[0];
        int outW = outSize[1];

        if (!Shape.strideDescendingCAscendingF(delta)) {
            // apparently not supported by cuDNN
            delta = delta.dup();
        }

        int[] srcStride = input.stride();
        int[] deltaStride = delta.stride();
        int[] algo1 = new int[1];
        int[] algo2 = new int[1];


        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.srcTensorDesc, dataType, miniBatch, inDepth, inH, inW,
                        srcStride[0], srcStride[1], srcStride[2], srcStride[3]));
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.deltaTensorDesc, dataType, miniBatch, outDepth, outH, outW,
                        deltaStride[0], deltaStride[1], deltaStride[2], deltaStride[3]));
        checkCudnn(cudnnSetConvolution2dDescriptor_v5(cudnnContext.convDesc, pad[0], pad[1], strides[0], strides[1], 1,
                        1, CUDNN_CROSS_CORRELATION, dataType));
        checkCudnn(cudnnSetFilter4dDescriptor(cudnnContext.filterDesc, dataType, tensorFormat, outDepth, inDepth, kH,
                        kW));
        if (mode == AlgoMode.USER_SPECIFIED && bwdFilterAlgo != null && bwdDataAlgo != null) {
            switch (bwdFilterAlgo) {
                case ALGO_0:
                    algo1[0] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
                    break;
                case ALGO_1:
                    algo1[0] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
                    break;
                case FFT:
                    algo1[0] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
                    break;
                case ALGO_3:
                    algo1[0] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
                    break;
                case WINOGRAD:
                    algo1[0] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD;
                    break;
                case WINOGRAD_NONFUSED:
                    algo1[0] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
                    break;
                case FFT_TILING:
                    algo1[0] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING;
                    break;
                case COUNT:
                    algo1[0] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
                    break;
                default:
                    throw new IllegalArgumentException("Unknown BwdFilterAlgo: " + bwdFilterAlgo);
            }

            switch (bwdDataAlgo) {
                case ALGO_0:
                    algo2[0] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
                    break;
                case ALGO_1:
                    algo2[0] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
                    break;
                case FFT:
                    algo2[0] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
                    break;
                case FFT_TILING:
                    algo2[0] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
                    break;
                case WINOGRAD:
                    algo2[0] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
                    break;
                case WINOGRAD_NONFUSED:
                    algo2[0] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
                    break;
                case COUNT:
                    algo2[0] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
                    break;
                default:
                    throw new IllegalArgumentException("Unknown BwdDataAlgo: " + bwdDataAlgo);
            }
        } else {
            checkCudnn(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnContext, cudnnContext.srcTensorDesc,
                            cudnnContext.deltaTensorDesc, cudnnContext.convDesc, cudnnContext.filterDesc,
                            mode == AlgoMode.NO_WORKSPACE ? CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE
                                            : CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                            0, algo1));
            checkCudnn(cudnnGetConvolutionBackwardDataAlgorithm(cudnnContext, cudnnContext.filterDesc,
                            cudnnContext.deltaTensorDesc, cudnnContext.convDesc, cudnnContext.srcTensorDesc,
                            mode == AlgoMode.NO_WORKSPACE ? CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE
                                            : CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                            0, algo2));
        }

        INDArray epsNext;
        if (Nd4j.getWorkspaceManager().checkIfWorkspaceExists(ComputationGraph.workspaceExternal)) {
            try (MemoryWorkspace workspace = Nd4j.getWorkspaceManager()
                            .getWorkspaceForCurrentThread(ComputationGraph.workspaceExternal).notifyScopeBorrowed()) {
                epsNext = Nd4j.create(new int[] {miniBatch, inDepth, inH, inW}, 'c');
            }
        } else
            epsNext = Nd4j.create(new int[] {miniBatch, inDepth, inH, inW}, 'c');

        int[] dstStride = epsNext.stride();

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareActionAllWrite(input, weights, weightGradView,
                        biasGradView, delta, epsNext);
        Pointer srcData = allocator.getPointer(input, context);
        Pointer filterData = allocator.getPointer(weights, context);
        Pointer filterGradData = allocator.getPointer(weightGradView, context);
        Pointer biasGradData = allocator.getPointer(biasGradView, context);
        Pointer deltaData = allocator.getPointer(delta, context);
        Pointer dstData = allocator.getPointer(epsNext, context);

        checkCudnn(cudnnSetStream(cudnnContext, new CUstream_st(context.getOldStream())));
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.dstTensorDesc, dataType, miniBatch, inDepth, inH, inW,
                        dstStride[0], dstStride[1], dstStride[2], dstStride[3]));
        checkCudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnContext, cudnnContext.srcTensorDesc,
                        cudnnContext.deltaTensorDesc, cudnnContext.convDesc, cudnnContext.filterDesc, algo1[0],
                        sizeInBytes));
        long sizeInBytes1 = sizeInBytes.get(0);
        checkCudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnContext, cudnnContext.filterDesc,
                        cudnnContext.deltaTensorDesc, cudnnContext.convDesc, cudnnContext.dstTensorDesc, algo2[0],
                        sizeInBytes));
        long sizeInBytes2 = sizeInBytes.get(0);
        if (sizeInBytes1 > workSpace.capacity() || sizeInBytes2 > workSpace.capacity()) {
            workSpace.deallocate();
            workSpace = new DataCache(Math.max(sizeInBytes1, sizeInBytes2));
        }

        checkCudnn(cudnnSetTensor4dDescriptor(cudnnContext.biasTensorDesc, tensorFormat, dataType, 1, outDepth, 1, 1));
        checkCudnn(cudnnConvolutionBackwardBias(cudnnContext, alpha, cudnnContext.deltaTensorDesc, deltaData, beta,
                        cudnnContext.biasTensorDesc, biasGradData));
        checkCudnn(cudnnConvolutionBackwardFilter(cudnnContext, alpha, cudnnContext.srcTensorDesc, srcData,
                        cudnnContext.deltaTensorDesc, deltaData, cudnnContext.convDesc, algo1[0], workSpace,
                        workSpace.capacity(), beta, cudnnContext.filterDesc, filterGradData));
        checkCudnn(cudnnConvolutionBackwardData(cudnnContext, alpha, cudnnContext.filterDesc, filterData,
                        cudnnContext.deltaTensorDesc, deltaData, cudnnContext.convDesc, algo2[0], workSpace,
                        workSpace.capacity(), beta, cudnnContext.dstTensorDesc, dstData));

        allocator.getFlowController().registerActionAllWrite(context, input, weights, weightGradView, biasGradView,
                        delta, epsNext);

        Gradient retGradient = new DefaultGradient();
        retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, biasGradView);
        retGradient.setGradientFor(ConvolutionParamInitializer.WEIGHT_KEY, weightGradView, 'c');

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            context.syncOldStream();

        return new Pair<>(retGradient, epsNext);
    }

    @Override
    public INDArray preOutput(INDArray input, INDArray weights, INDArray bias, int[] kernel, int[] strides, int[] pad,
                    AlgoMode mode, FwdAlgo fwdAlgo, ConvolutionMode convolutionMode) {
        int miniBatch = input.size(0);
        int inH = input.size(2);
        int inW = input.size(3);

        int outDepth = weights.size(0);
        int inDepth = weights.size(1);
        int kH = weights.size(2);
        int kW = weights.size(3);

        int[] srcStride = input.stride();

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, null, convolutionMode); //Also performs validation
            pad = ConvolutionUtils.getSameModeBottomRightPadding(outSize, new int[] {inH, inW}, kernel, strides);
        } else {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, pad, convolutionMode); //Also performs validation
        }


        INDArray z;

        if (Nd4j.getWorkspaceManager().checkIfWorkspaceExists(ComputationGraph.workspaceExternal)) {
            try (MemoryWorkspace workspace = Nd4j.getWorkspaceManager()
                            .getWorkspaceForCurrentThread(ComputationGraph.workspaceExternal).notifyScopeBorrowed()) {
                z = Nd4j.createUninitialized(new int[] {miniBatch, outDepth, outSize[0], outSize[1]});
            }
        } else
            z = Nd4j.createUninitialized(new int[] {miniBatch, outDepth, outSize[0], outSize[1]});

        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.srcTensorDesc, dataType, miniBatch, inDepth, inH, inW,
                        srcStride[0], srcStride[1], srcStride[2], srcStride[3]));
        checkCudnn(cudnnSetFilter4dDescriptor(cudnnContext.filterDesc, dataType, tensorFormat, outDepth, inDepth, kH,
                        kW));
        checkCudnn(cudnnSetConvolution2dDescriptor_v5(cudnnContext.convDesc, pad[0], pad[1], strides[0], strides[1], 1,
                        1, CUDNN_CROSS_CORRELATION, dataType));

        // find dimension of convolution output
        //        checkCudnn(cudnnGetConvolution2dForwardOutputDim(cudnnContext.convDesc, cudnnContext.srcTensorDesc, cudnnContext.filterDesc, n, c, h, w));
        //        INDArray z = Nd4j.createUninitialized(new int[]{n[0],c[0],h[0],w[0]},'c');


        int[] algo = new int[1];
        int[] dstStride = z.stride();
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.dstTensorDesc, dataType, miniBatch, outDepth, outSize[0],
                        outSize[1], dstStride[0], dstStride[1], dstStride[2], dstStride[3]));
        if (mode == AlgoMode.USER_SPECIFIED && fwdAlgo != null) {
            switch (fwdAlgo) {
                case IMPLICIT_GEMM:
                    algo[0] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
                    break;
                case IMPLICIT_PRECOMP_GEMM:
                    algo[0] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
                    break;
                case GEMM:
                    algo[0] = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
                    break;
                case DIRECT:
                    algo[0] = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
                    break;
                case FFT:
                    algo[0] = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
                    break;
                case FFT_TILING:
                    algo[0] = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
                    break;
                case WINOGRAD:
                    algo[0] = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
                    break;
                case WINOGRAD_NONFUSED:
                    algo[0] = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
                    break;
                case COUNT:
                    algo[0] = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
                    break;
                default:
                    throw new IllegalArgumentException("Unknown FwdAlgo: " + fwdAlgo);
            }
        } else {
            checkCudnn(cudnnGetConvolutionForwardAlgorithm(cudnnContext, cudnnContext.srcTensorDesc,
                            cudnnContext.filterDesc, cudnnContext.convDesc,
                            cudnnContext.dstTensorDesc, mode == AlgoMode.NO_WORKSPACE
                                            ? CUDNN_CONVOLUTION_FWD_NO_WORKSPACE : CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                            0, algo));
        }

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareAction(z, input, weights, bias);
        Pointer srcData = allocator.getPointer(input, context);
        Pointer filterData = allocator.getPointer(weights, context);
        Pointer biasData = allocator.getPointer(bias, context);
        Pointer dstData = allocator.getPointer(z, context);

        checkCudnn(cudnnSetStream(cudnnContext, new CUstream_st(context.getOldStream())));
        checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(cudnnContext, cudnnContext.srcTensorDesc,
                        cudnnContext.filterDesc, cudnnContext.convDesc, cudnnContext.dstTensorDesc, algo[0],
                        sizeInBytes));
        if (sizeInBytes.get(0) > workSpace.capacity()) {
            workSpace.deallocate();
            workSpace = new DataCache(sizeInBytes.get(0));
        }
        checkCudnn(cudnnConvolutionForward(cudnnContext, alpha, cudnnContext.srcTensorDesc, srcData,
                        cudnnContext.filterDesc, filterData, cudnnContext.convDesc, algo[0], workSpace,
                        workSpace.capacity(), beta, cudnnContext.dstTensorDesc, dstData));

        checkCudnn(cudnnSetTensor4dDescriptor(cudnnContext.biasTensorDesc, tensorFormat, dataType, 1, outDepth, 1, 1));
        checkCudnn(cudnnAddTensor(cudnnContext, alpha, cudnnContext.biasTensorDesc, biasData, alpha,
                        cudnnContext.dstTensorDesc, dstData));

        allocator.registerAction(context, z, input, weights, bias);

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            context.syncOldStream();

        return z;
    }

    @Override
    public INDArray activate(INDArray z, IActivation afn) {
        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        INDArray activation = z;

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareAction(z);
        Pointer dstData = allocator.getPointer(z, context);

        checkCudnn(cudnnSetStream(cudnnContext, new CUstream_st(context.getOldStream())));
        switch (afn.toString()) {
            case "identity":
                break;
            case "sigmoid":
                checkCudnn(cudnnSetActivationDescriptor(cudnnContext.activationDesc, CUDNN_ACTIVATION_SIGMOID,
                                CUDNN_PROPAGATE_NAN, 0));
                checkCudnn(cudnnActivationForward(cudnnContext, cudnnContext.activationDesc, alpha,
                                cudnnContext.dstTensorDesc, dstData, beta, cudnnContext.dstTensorDesc, dstData));
                break;
            case "relu":
                checkCudnn(cudnnSetActivationDescriptor(cudnnContext.activationDesc, CUDNN_ACTIVATION_RELU,
                                CUDNN_PROPAGATE_NAN, 0));
                checkCudnn(cudnnActivationForward(cudnnContext, cudnnContext.activationDesc, alpha,
                                cudnnContext.dstTensorDesc, dstData, beta, cudnnContext.dstTensorDesc, dstData));
                break;
            case "tanh":
                checkCudnn(cudnnSetActivationDescriptor(cudnnContext.activationDesc, CUDNN_ACTIVATION_TANH,
                                CUDNN_PROPAGATE_NAN, 0));
                checkCudnn(cudnnActivationForward(cudnnContext, cudnnContext.activationDesc, alpha,
                                cudnnContext.dstTensorDesc, dstData, beta, cudnnContext.dstTensorDesc, dstData));
                break;
            case "softmax":
                checkCudnn(cudnnSoftmaxForward(cudnnContext, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, alpha,
                                cudnnContext.dstTensorDesc, dstData, beta, cudnnContext.dstTensorDesc, dstData));
                break;
            case "logsoftmax":
                checkCudnn(cudnnSoftmaxForward(cudnnContext, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL, alpha,
                                cudnnContext.dstTensorDesc, dstData, beta, cudnnContext.dstTensorDesc, dstData));
                break;
            default:
                activation = null;
        }

        allocator.registerAction(context, activation);

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            context.syncOldStream();

        return activation;
    }

}
