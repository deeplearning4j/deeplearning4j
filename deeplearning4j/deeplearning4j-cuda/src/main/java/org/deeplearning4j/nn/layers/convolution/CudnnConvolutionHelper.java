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

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.BwdDataAlgo;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.BwdFilterAlgo;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.FwdAlgo;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseCudnnHelper;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.nd4j.util.OneTimeLogger;
import org.nd4j.util.StringUtils;

import java.util.Arrays;

import static org.bytedeco.javacpp.cuda.CUstream_st;
import static org.bytedeco.javacpp.cudnn.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

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
                    ConvolutionMode convolutionMode, int[] dilation, LayerWorkspaceMgr workspaceMgr) {
        if(dilation[0] > 2 || dilation[1] > 2){
            //CuDNN seems to not support all (valid) configurations...
            //Same mode + dilation 3: cuDNN status = 9: CUDNN_STATUS_NOT_SUPPORTED
            return null;
        }
        int code;

        val miniBatch = input.size(0);
        val outDepth = weights.size(0);
        val inDepth = weights.size(1);
        val kH = weights.size(2);
        val kW = weights.size(3);

        CudnnForwardArgs args = getCudnnForwardArgs(input, kernel, strides, pad, dilation, convolutionMode);
        input = args.getInput();
        val inH = input.size(2);
        val inW = input.size(3);
        val srcStride = input.stride();
        val outSize = args.getOutSize();
        val outH = outSize[0];
        val outW = outSize[1];

        if (!Shape.strideDescendingCAscendingF(delta)) {
            // apparently not supported by cuDNN
            delta = delta.dup();
        }

        val deltaStride = delta.stride();
        int[] algo1 = new int[1];
        int[] algo2 = new int[1];


        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        code = cudnnSetTensor4dDescriptorEx(cudnnContext.srcTensorDesc, dataType, (int) miniBatch, (int) inDepth,(int)  inH, (int) inW,
                (int) srcStride[0], (int) srcStride[1], (int) srcStride[2], (int) srcStride[3]);
        checkCudnn(false, "cudnnSetTensor4dDescriptorEx", code, input, weights, null, delta, kernel, strides, pad, mode, null, bwdFilterAlgo, bwdDataAlgo, convolutionMode, dilation);
        code = cudnnSetTensor4dDescriptorEx(cudnnContext.deltaTensorDesc, dataType, (int) miniBatch, (int) outDepth, (int) outH, (int) outW,
                (int) deltaStride[0], (int) deltaStride[1], (int) deltaStride[2], (int) deltaStride[3]);
        checkCudnn(false, "cudnnSetTensor4dDescriptorEx", code, input, weights, null, delta, kernel, strides, pad, mode, null, bwdFilterAlgo, bwdDataAlgo, convolutionMode, dilation);
        code = cudnnSetConvolution2dDescriptor(cudnnContext.convDesc, pad[0], pad[1], strides[0], strides[1], dilation[0],
                        dilation[1], CUDNN_CROSS_CORRELATION, dataType);
        checkCudnn(false, "cudnnSetConvolution2dDescriptor", code, input, weights, null, delta, kernel, strides, pad, mode, null, bwdFilterAlgo, bwdDataAlgo, convolutionMode, dilation);
        code = cudnnSetFilter4dDescriptor(cudnnContext.filterDesc, dataType, TENSOR_FORMAT, (int) outDepth, (int) inDepth, (int) kH, (int) kW);
        checkCudnn(false, "cudnnSetFilter4dDescriptor", code, input, weights, null, delta, kernel, strides, pad, mode, null, bwdFilterAlgo, bwdDataAlgo, convolutionMode, dilation);

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
            code = cudnnGetConvolutionBackwardFilterAlgorithm(cudnnContext, cudnnContext.srcTensorDesc,
                            cudnnContext.deltaTensorDesc, cudnnContext.convDesc, cudnnContext.filterDesc,
                            mode == AlgoMode.NO_WORKSPACE ? CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE
                                            : CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                            0, algo1);
            checkCudnn(false, "cudnnGetConvolutionBackwardFilterAlgorithm", code, input, weights, null, delta, kernel, strides, pad, mode, null, bwdFilterAlgo, bwdDataAlgo, convolutionMode, dilation);
            code = cudnnGetConvolutionBackwardDataAlgorithm(cudnnContext, cudnnContext.filterDesc,
                            cudnnContext.deltaTensorDesc, cudnnContext.convDesc, cudnnContext.srcTensorDesc,
                            mode == AlgoMode.NO_WORKSPACE ? CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE
                                            : CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                            0, algo2);
            checkCudnn(false, "cudnnGetConvolutionBackwardDataAlgorithm", code, input, weights, null, delta, kernel, strides, pad, mode, null, bwdFilterAlgo, bwdDataAlgo, convolutionMode, dilation);
        }

        if(log.isTraceEnabled()){
            BwdFilterAlgo fa = BwdFilterAlgo.values()[algo1[0]];
            BwdDataAlgo da = BwdDataAlgo.values()[algo2[0]];
            log.trace("CudnnConvolutionHelper backward algorithm selection: mode {}, filter algorithm {}, data algorithm {}", mode, fa, da);
        }

        INDArray epsNext = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, new int[] {(int) miniBatch,(int)  inDepth, (int) inH, (int) inW}, 'c');

        val dstStride = epsNext.stride();

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareActionAllWrite(input, weights, weightGradView,
                        biasGradView, delta, epsNext);
        Pointer srcData = allocator.getPointer(input, context);
        Pointer filterData = allocator.getPointer(weights, context);
        Pointer filterGradData = allocator.getPointer(weightGradView, context);
        Pointer biasGradData = allocator.getPointer(biasGradView, context);
        Pointer deltaData = allocator.getPointer(delta, context);
        Pointer dstData = allocator.getPointer(epsNext, context);

        code = cudnnSetStream(cudnnContext, new CUstream_st(context.getOldStream()));
        checkCudnn(false, "cudnnSetStream", code, input, weights, null, delta, kernel, strides, pad, mode, null, bwdFilterAlgo, bwdDataAlgo, convolutionMode, dilation);

        code = cudnnSetTensor4dDescriptorEx(cudnnContext.dstTensorDesc, dataType, (int) miniBatch, (int) inDepth, (int) inH, (int) inW,
                (int) dstStride[0], (int) dstStride[1], (int) dstStride[2], (int) dstStride[3]);
        checkCudnn(false, "cudnnSetTensor4dDescriptorEx", code, input, weights, null, delta, kernel, strides, pad, mode, null, bwdFilterAlgo, bwdDataAlgo, convolutionMode, dilation);

        code = cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnContext, cudnnContext.srcTensorDesc,
                        cudnnContext.deltaTensorDesc, cudnnContext.convDesc, cudnnContext.filterDesc, algo1[0],
                        sizeInBytes);
        checkCudnn(false, "cudnnGetConvolutionBackwardFilterWorkspaceSize", code, input, weights, null, delta, kernel, strides, pad, mode, null, bwdFilterAlgo, bwdDataAlgo, convolutionMode, dilation);

        long sizeInBytes1 = sizeInBytes.get(0);
        code = cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnContext, cudnnContext.filterDesc,
                        cudnnContext.deltaTensorDesc, cudnnContext.convDesc, cudnnContext.dstTensorDesc, algo2[0],
                        sizeInBytes);
        checkCudnn(false, "cudnnGetConvolutionBackwardDataWorkspaceSize", code, input, weights, null, delta, kernel, strides, pad, mode, null, bwdFilterAlgo, bwdDataAlgo, convolutionMode, dilation);

        long sizeInBytes2 = sizeInBytes.get(0);
        if (sizeInBytes1 > workSpace.capacity() || sizeInBytes2 > workSpace.capacity()) {
            long newSize = Math.max(sizeInBytes1, sizeInBytes2);
            if(log.isTraceEnabled()){
                log.trace("CudnnConvolutionHelper: Deallocating workspace of size {} ({}), allocating new workspace of size {} ({})",
                        workSpace.capacity(), StringUtils.TraditionalBinaryPrefix.long2String(workSpace.capacity(), null, 2),
                        newSize, StringUtils.TraditionalBinaryPrefix.long2String(newSize, null, 2));
            }
            workSpace.deallocate();
            workSpace = new DataCache(newSize);
        }

        code = cudnnSetTensor4dDescriptor(cudnnContext.biasTensorDesc, TENSOR_FORMAT, dataType, 1, (int) outDepth, 1, 1);
        checkCudnn(false, "cudnnSetTensor4dDescriptor", code, input, weights, null, delta, kernel, strides, pad, mode, null, bwdFilterAlgo, bwdDataAlgo, convolutionMode, dilation);

        code = cudnnConvolutionBackwardBias(cudnnContext, alpha, cudnnContext.deltaTensorDesc, deltaData, beta,
                        cudnnContext.biasTensorDesc, biasGradData);
        checkCudnn(false, "cudnnConvolutionBackwardBias", code, input, weights, null, delta, kernel, strides, pad, mode, null, bwdFilterAlgo, bwdDataAlgo, convolutionMode, dilation);

        code = cudnnConvolutionBackwardFilter(cudnnContext, alpha, cudnnContext.srcTensorDesc, srcData,
                        cudnnContext.deltaTensorDesc, deltaData, cudnnContext.convDesc, algo1[0], workSpace,
                        workSpace.capacity(), beta, cudnnContext.filterDesc, filterGradData);
        checkCudnn(false, "cudnnConvolutionBackwardFilter", code, input, weights, null, delta, kernel, strides, pad, mode, null, bwdFilterAlgo, bwdDataAlgo, convolutionMode, dilation);

        code = cudnnConvolutionBackwardData(cudnnContext, alpha, cudnnContext.filterDesc, filterData,
                        cudnnContext.deltaTensorDesc, deltaData, cudnnContext.convDesc, algo2[0], workSpace,
                        workSpace.capacity(), beta, cudnnContext.dstTensorDesc, dstData);
        checkCudnn(false, "cudnnConvolutionBackwardData", code, input, weights, null, delta, kernel, strides, pad, mode, null, bwdFilterAlgo, bwdDataAlgo, convolutionMode, dilation);

        allocator.getFlowController().registerActionAllWrite(context, input, weights, weightGradView, biasGradView,
                        delta, epsNext);

        Gradient retGradient = new DefaultGradient();
        retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, biasGradView);
        retGradient.setGradientFor(ConvolutionParamInitializer.WEIGHT_KEY, weightGradView, 'c');

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            context.syncOldStream();

        //Note that: if we had to manually pad for SAME mode, we have to 'undo' this manual padding for the epsilon
        // we return. The returned epsilon (i.e., dL/dIn array) has to be the same shape as the *original* input.
        if(args.isManualPadBottom() || args.isManualPadRight()) {
            epsNext = epsNext.get(all(), all(),
                    interval(0, epsNext.size(2) - (args.isManualPadBottom() ? 1 : 0)),
                    interval(0, epsNext.size(3) - (args.isManualPadRight() ? 1 : 0)));
        }

        return new Pair<>(retGradient, epsNext);
    }

    @Override
    public INDArray preOutput(INDArray input, INDArray weights, INDArray bias, int[] kernel, int[] strides, int[] pad,
                              AlgoMode mode, FwdAlgo fwdAlgo, ConvolutionMode convolutionMode, int[] dilation, LayerWorkspaceMgr workspaceMgr) {
        if(dilation[0] > 2 || dilation[1] > 2){
            //CuDNN seems to not support all (valid) configurations...
            //Same mode + dilation 3: cuDNN status = 9: CUDNN_STATUS_NOT_SUPPORTED
            return null;
        }
        int code;

        val miniBatch = input.size(0);
        val outDepth = weights.size(0);
        val inDepth = weights.size(1);
        val kH = weights.size(2);
        val kW = weights.size(3);

        CudnnForwardArgs args = getCudnnForwardArgs(input, kernel, strides, pad, dilation, convolutionMode);
        input = args.getInput();
        val inH = input.size(2);
        val inW = input.size(3);
        val srcStride = input.stride();
        val outSize = args.getOutSize();

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        INDArray z = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new int[] {(int) miniBatch, (int) outDepth, outSize[0], outSize[1]});

        code = cudnnSetTensor4dDescriptorEx(cudnnContext.srcTensorDesc, dataType, (int) miniBatch, (int) inDepth, (int) inH, (int) inW,
                (int)  srcStride[0], (int) srcStride[1], (int) srcStride[2], (int) srcStride[3]);
        checkCudnn(true, "cudnnSetTensor4dDescriptorEx", code, input, weights, bias, null, kernel, strides, pad, mode, fwdAlgo, null, null, convolutionMode, dilation);

        code = cudnnSetFilter4dDescriptor(cudnnContext.filterDesc, dataType, TENSOR_FORMAT, (int) outDepth, (int) inDepth, (int) kH, (int) kW);
        checkCudnn(true, "cudnnSetFilter4dDescriptor", code, input, weights, bias, null, kernel, strides, pad, mode, fwdAlgo, null, null, convolutionMode, dilation);

        code = cudnnSetConvolution2dDescriptor(cudnnContext.convDesc, pad[0], pad[1], strides[0], strides[1], dilation[0],
                        dilation[1], CUDNN_CROSS_CORRELATION, dataType);
        checkCudnn(true, "cudnnSetConvolution2dDescriptor", code, input, weights, bias, null, kernel, strides, pad, mode, fwdAlgo, null, null, convolutionMode, dilation);


        // find dimension of convolution output
        //        checkCudnn(cudnnGetConvolution2dForwardOutputDim(cudnnContext.convDesc, cudnnContext.srcTensorDesc, cudnnContext.filterDesc, n, c, h, w));
        //        INDArray z = Nd4j.createUninitialized(new int[]{n[0],c[0],h[0],w[0]},'c');


        int[] algo = new int[1];
        val dstStride = z.stride();
        code = cudnnSetTensor4dDescriptorEx(cudnnContext.dstTensorDesc, dataType, (int) miniBatch, (int) outDepth, (int) outSize[0],
                (int) outSize[1], (int) dstStride[0], (int) dstStride[1], (int) dstStride[2], (int) dstStride[3]);
        checkCudnn(true, "cudnnSetTensor4dDescriptorEx", code, input, weights, bias, null, kernel, strides, pad, mode, fwdAlgo, null, null, convolutionMode, dilation);

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
            code = cudnnGetConvolutionForwardAlgorithm(cudnnContext, cudnnContext.srcTensorDesc,
                    cudnnContext.filterDesc, cudnnContext.convDesc,
                    cudnnContext.dstTensorDesc, mode == AlgoMode.NO_WORKSPACE
                            ? CUDNN_CONVOLUTION_FWD_NO_WORKSPACE : CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                    0, algo);

            if(code != CUDNN_STATUS_SUCCESS){
                //If CuDNN can't infer algorithm - try IMPLICIT_GEMM
                //Why this specifically? According to the docs, it seems to have the least number of restrictions
                // to things like dilation

                OneTimeLogger.warn(log, "Error getting CuDNN forward algorithm - falling back on IMPLICIT_GEMM");
                mode = AlgoMode.USER_SPECIFIED;
                fwdAlgo = FwdAlgo.IMPLICIT_GEMM;
                algo[0] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            }
        }

        if(log.isTraceEnabled()){
            FwdAlgo a = FwdAlgo.values()[algo[0]];
            log.trace("CudnnConvolutionHelper forward algorithm selection: mode {}, algorithm {}", mode, a);
        }

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareAction(z, input, weights, bias);
        Pointer srcData = allocator.getPointer(input, context);
        Pointer filterData = allocator.getPointer(weights, context);
        Pointer biasData = allocator.getPointer(bias, context);
        Pointer dstData = allocator.getPointer(z, context);

        code = cudnnSetStream(cudnnContext, new CUstream_st(context.getOldStream()));
        checkCudnn(true, "cudnnSetStream", code, input, weights, bias, null, kernel, strides, pad, mode, fwdAlgo, null, null, convolutionMode, dilation);

        code = cudnnGetConvolutionForwardWorkspaceSize(cudnnContext, cudnnContext.srcTensorDesc,
                        cudnnContext.filterDesc, cudnnContext.convDesc, cudnnContext.dstTensorDesc, algo[0],
                        sizeInBytes);
        checkCudnn(true, "cudnnGetConvolutionForwardWorkspaceSize", code, input, weights, bias, null, kernel, strides, pad, mode, fwdAlgo, null, null, convolutionMode, dilation);

        if (sizeInBytes.get(0) > workSpace.capacity()) {
            workSpace.deallocate();
            workSpace = new DataCache(sizeInBytes.get(0));
        }
        code = cudnnConvolutionForward(cudnnContext, alpha, cudnnContext.srcTensorDesc, srcData,
                        cudnnContext.filterDesc, filterData, cudnnContext.convDesc, algo[0], workSpace,
                        workSpace.capacity(), beta, cudnnContext.dstTensorDesc, dstData);
        checkCudnn(true, "cudnnConvolutionForward", code, input, weights, bias, null, kernel, strides, pad, mode, fwdAlgo, null, null, convolutionMode, dilation);


        code = cudnnSetTensor4dDescriptor(cudnnContext.biasTensorDesc, TENSOR_FORMAT, dataType, 1, (int) outDepth, 1, 1);
        checkCudnn(true, "cudnnSetTensor4dDescriptor", code, input, weights, bias, null, kernel, strides, pad, mode, fwdAlgo, null, null, convolutionMode, dilation);

        code = cudnnAddTensor(cudnnContext, alpha, cudnnContext.biasTensorDesc, biasData, alpha,
                        cudnnContext.dstTensorDesc, dstData);
        checkCudnn(true, "cudnnAddTensor", code, input, weights, bias, null, kernel, strides, pad, mode, fwdAlgo, null, null, convolutionMode, dilation);

        allocator.registerAction(context, z, input, weights, bias);

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            context.syncOldStream();

        return z;
    }

    private void checkCudnn(boolean forward, String step, int code, INDArray input, INDArray weights, INDArray bias, INDArray delta,
                            int[] kernel, int[] strides, int[] pad,
                            AlgoMode mode, FwdAlgo fwdAlgo, BwdFilterAlgo bwdFilterAlgo, BwdDataAlgo bwdDataAlgo, ConvolutionMode convolutionMode, int[] dilation) {

        if (code != CUDNN_STATUS_SUCCESS) {
            StringBuilder sb = new StringBuilder();
            sb.append("CuDNN error = ").append(code).append(": ").append(cudnnGetErrorString(code).getString())
                    .append(" during ")
                    .append(forward ? "forward pass" : "backward pass")
                    .append(" - step ").append(step)
                    .append(": inputShape=").append(Arrays.toString(input.shape()))
                    .append(", weightsShape=").append(Arrays.toString(weights.shape()))
                    .append(", biasShape=").append(bias == null ? null : Arrays.toString(bias.shape()));
            if (!forward) {
                sb.append(", gradientShape=").append(Arrays.toString(delta.shape()));
            }
            sb.append(", kernel=").append(Arrays.toString(kernel))
                    .append(", stride=").append(Arrays.toString(strides))
                    .append(", padding=").append(Arrays.toString(pad))
                    .append(", dilation=").append(Arrays.toString(dilation))
                    .append(", AlgoMode=").append(mode);
            if (forward) {
                sb.append(", fwdAlgo=").append(fwdAlgo);
            } else {
                sb.append(", bwdFilterAlgo=").append(bwdFilterAlgo)
                        .append(", bwdDataAlgo=").append(bwdDataAlgo);
            }
            sb.append(", convolutionMode=").append(convolutionMode);

            throw new RuntimeException(sb.toString());
        }
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

    public static CudnnForwardArgs getCudnnForwardArgs(INDArray input, int[] kernel, int[] strides, int[] padding, int[] dilation,
                                                   ConvolutionMode convolutionMode){
        INDArray origInput = input;

        //Check if we need to dup the input: views, non-contiguous, etc. CuDNN also seems to have has issues if strides
        // are non-default for C order - even if they *should* be OK otherwise
        if(input.isView() || !Shape.hasDefaultStridesForShape(input)){
            input = input.dup('c');
        }


        val inH = input.size(2);
        val inW = input.size(3);

        boolean manualPadBottom = false;
        boolean manualPadRight = false;

        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, null, convolutionMode, dilation); //Also performs validation
            padding = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new int[] {(int) inH, (int) inW}, kernel, strides, dilation);
            int[] padBottomRight = ConvolutionUtils.getSameModeBottomRightPadding(outSize, new int[] {(int) inH, (int) inW}, kernel, strides, dilation);
            if(!Arrays.equals(padding, padBottomRight)){
                /*
                CuDNN - even as of 7.1 (CUDA 9.1) still doesn't have support for proper SAME mode padding (i.e., asymmetric
                padding) - padding can *only* be specified as the same amount for both the top/bottom, and for left/right.
                In SAME mode padding, sometimes these are the same - but often they are not.
                Note that when they differ, the bottom or right padding will be exactly 1 more than the top or left padding.
                As per TF, we'll manually pad here: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/conv_ops.cc#L571-L607
                 */
                manualPadBottom = (padding[0] != padBottomRight[0]);
                manualPadRight = (padding[1] != padBottomRight[1]);

                //NCHW format
                val newShape = new long[]{input.size(0), input.size(1),
                        input.size(2) + (manualPadBottom ? 1 : 0),
                        input.size(3) + (manualPadRight ? 1 : 0)};
                INDArray newInput = Nd4j.create(newShape);
                newInput.put(new INDArrayIndex[]{all(), all(), interval(0,input.size(2)),
                        interval(0, input.size(3))}, input);
                input = newInput;
                //Now: we've manually applied the "extra" bottom/right padding only - if required. Consequently, we
                // now have the same amount of padding required for top/bottom, and left/right - which we'll let
                // CuDNN handle
            }
        } else {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, padding, convolutionMode, dilation); //Also performs validation
        }

        return new CudnnForwardArgs(manualPadBottom, manualPadRight, input, origInput, padding, outSize);
    }


    @AllArgsConstructor
    @Data
    public static class CudnnForwardArgs {
        private boolean manualPadBottom;
        private boolean manualPadRight;
        private INDArray input;
        private INDArray origInput;
        private int[] padding;
        private int[] outSize;
    }

}
