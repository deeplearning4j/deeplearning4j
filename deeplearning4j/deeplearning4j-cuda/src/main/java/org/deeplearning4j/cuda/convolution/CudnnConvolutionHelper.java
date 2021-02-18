/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.cuda.convolution;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import com.jakewharton.byteunits.BinaryByteUnit;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.BwdDataAlgo;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.BwdFilterAlgo;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.FwdAlgo;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.cuda.BaseCudnnHelper;
import org.deeplearning4j.nn.layers.convolution.ConvolutionHelper;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.nd4j.common.util.OneTimeLogger;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import org.bytedeco.cuda.cudart.*;
import org.bytedeco.cuda.cudnn.*;

import static org.bytedeco.cuda.global.cudnn.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/**
 * cuDNN-based helper for the convolution layer.
 *
 * @author saudet
 */
@Slf4j
public class CudnnConvolutionHelper extends BaseCudnnHelper implements ConvolutionHelper {

    public CudnnConvolutionHelper(DataType dataType) {
        super(dataType);
    }

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

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray weights, INDArray bias, INDArray delta, int[] kernel,
                                                     int[] strides, int[] pad, INDArray biasGradView, INDArray weightGradView, IActivation afn,
                                                     AlgoMode mode, BwdFilterAlgo bwdFilterAlgo, BwdDataAlgo bwdDataAlgo,
                                                     ConvolutionMode convolutionMode, int[] dilation, CNN2DFormat format, LayerWorkspaceMgr workspaceMgr) {

        //AB 2020/04/21 - cuDNN does have NHWC support (with limitations) however I have been unable to get it working
        // correctly on NHWC data, even after updating all descriptors, tensor format, etc.
        //Therefore: all computation here is done in NCHW format only
        //As of a future (next?) release we'll likely switch to C++ for cuDNN support
        boolean origNHWC = false;
        if(format == CNN2DFormat.NHWC){
            input = input.permute(0,3,1,2); //NHWC to NCHW
            delta = delta.permute(0,3,1,2);
            origNHWC = true;
        }

        int TENSOR_FORMAT = CUDNN_TENSOR_NCHW;

        int code;

        val miniBatch = input.size(0);
        val outDepth = weights.size(0);
        val inDepth = weights.size(1);
        val kH = weights.size(2);
        val kW = weights.size(3);

        CudnnForwardArgs args = getCudnnForwardArgs(input, kernel, strides, pad, dilation, convolutionMode, null, CNN2DFormat.NCHW);    //Note hardcoded NCHW due to above
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
            /*
            code = cudnnGetConvolutionBackwardFilterAlgorithm(cudnnContext, cudnnContext.srcTensorDesc,
                    cudnnContext.deltaTensorDesc, cudnnContext.convDesc, cudnnContext.filterDesc,
                    mode == AlgoMode.NO_WORKSPACE ? CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE
                            : CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                    0, algo1);
            */
            val fa = new cudnnConvolutionBwdFilterAlgoPerf_t();
            val counts = new int[1];
            code = cudnnFindConvolutionBackwardFilterAlgorithm(cudnnContext,  cudnnContext.srcTensorDesc,
                    cudnnContext.deltaTensorDesc, cudnnContext.convDesc, cudnnContext.filterDesc, 1, counts, fa);
            algo1[0] = fa.algo();

            checkCudnn(false, "cudnnGetConvolutionBackwardFilterAlgorithm", code, input, weights, null, delta, kernel, strides, pad, mode, null, bwdFilterAlgo, bwdDataAlgo, convolutionMode, dilation);

            /*
            code = cudnnGetConvolutionBackwardDataAlgorithm(cudnnContext, cudnnContext.filterDesc,
                    cudnnContext.deltaTensorDesc, cudnnContext.convDesc, cudnnContext.srcTensorDesc,
                    mode == AlgoMode.NO_WORKSPACE ? CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE
                            : CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                    0, algo2);
             */

            val da = new cudnnConvolutionBwdDataAlgoPerf_t();
            code = cudnnFindConvolutionBackwardDataAlgorithm(cudnnContext, cudnnContext.filterDesc,
                    cudnnContext.deltaTensorDesc, cudnnContext.convDesc, cudnnContext.srcTensorDesc, 1, counts, da);

            algo2[0] = da.algo();
            checkCudnn(false, "cudnnGetConvolutionBackwardDataAlgorithm", code, input, weights, null, delta, kernel, strides, pad, mode, null, bwdFilterAlgo, bwdDataAlgo, convolutionMode, dilation);
        }

        if(log.isTraceEnabled()){
            BwdFilterAlgo fa = BwdFilterAlgo.values()[algo1[0]];
            BwdDataAlgo da = BwdDataAlgo.values()[algo2[0]];
            log.trace("CudnnConvolutionHelper backward algorithm selection: mode {}, filter algorithm {}, data algorithm {}", mode, fa, da);
        }

        INDArray epsNext = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, weights.dataType(), new long[] {(int) miniBatch,(int)  inDepth, (int) inH, (int) inW}, 'c');

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

        code = cudnnSetStream(cudnnContext, new CUstream_st(context.getCublasStream()));
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

        DataCache workSpace = workspaceMgr.getHelperWorkspace(LayerWorkspaceMgr.CUDNN_WORKSPACE_KEY);
        long sizeInBytes2 = sizeInBytes.get(0);
        if (workSpace == null || sizeInBytes1 > workSpace.capacity() || sizeInBytes2 > workSpace.capacity()) {
            long newSize = Math.max(sizeInBytes1, sizeInBytes2);
            if(log.isTraceEnabled()){
                if(workSpace == null){
                    log.trace("CudnnConvolutionHelper backpropGradient: Allocating initial workspace of size {} ({})", newSize,
                            BinaryByteUnit.format(newSize, "#.00"));
                } else {
                    log.trace("CudnnConvolutionHelper backpropGradient: Deallocating workspace of size {} ({}), allocating new workspace of size {} ({})",
                            workSpace.capacity(), BinaryByteUnit.format(workSpace.capacity(), "#.00"),
                            newSize, BinaryByteUnit.format(newSize, "#.00"));
                }
            }
            if(workSpace != null)
                workSpace.deallocate();
            workSpace = new DataCache(newSize);
            workspaceMgr.setHelperWorkspace(LayerWorkspaceMgr.CUDNN_WORKSPACE_KEY, workSpace);
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

        if(origNHWC){
            epsNext = epsNext.permute(0,2,3,1);     //NCHW to NHWC
        }

        return new Pair<>(retGradient, epsNext);
    }

    @Override
    public INDArray preOutput(INDArray input, INDArray weights, INDArray bias, int[] kernel, int[] strides, int[] pad,
                              AlgoMode mode, FwdAlgo fwdAlgo, ConvolutionMode convolutionMode, int[] dilation, CNN2DFormat format,
                              LayerWorkspaceMgr workspaceMgr) {

        //AB 2020/04/21 - cuDNN does have NHWC support (with limitations) however I have been unable to get it working
        // correctly on NHWC data, even after updating all descriptors, tensor format, etc.
        //Therefore: all computation here is done in NCHW format only
        //As of a future (next?) release we'll likely switch to C++ for cuDNN support
        boolean origNHWC = false;
        if(format == CNN2DFormat.NHWC){
            input = input.permute(0,3,1,2); //NHWC to NCHW
            origNHWC = true;
        }

        int TENSOR_FORMAT = CUDNN_TENSOR_NCHW;

        int code;

        val miniBatch = input.size(0);
        val outDepth = weights.size(0);
        val inDepth = weights.size(1);
        val kH = weights.size(2);
        val kW = weights.size(3);

        CudnnForwardArgs args = getCudnnForwardArgs(input, kernel, strides, pad, dilation, convolutionMode, null, CNN2DFormat.NCHW);        //Note hardcoded NCHW due to above
        input = args.getInput();
        val inH = input.size(2);
        val inW = input.size(3);
        val srcStride = input.stride();
        val outSize = args.getOutSize();

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        INDArray z = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, weights.dataType(), new long[] {(int) miniBatch, (int) outDepth, outSize[0], outSize[1]});

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
            /*
            code = cudnnGetConvolutionForwardAlgorithm_v7(cudnnContext, cudnnContext.srcTensorDesc,
                    cudnnContext.filterDesc, cudnnContext.convDesc,
                    cudnnContext.dstTensorDesc, mode == AlgoMode.NO_WORKSPACE
                            ? CUDNN_CONVOLUTION_FWD_ : CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                    0, algo);
            */

            val cdf = new cudnnConvolutionFwdAlgoPerf_t();
            val count = new int[1];
            code = cudnnFindConvolutionForwardAlgorithm(cudnnContext, cudnnContext.srcTensorDesc, cudnnContext.filterDesc, cudnnContext.convDesc, cudnnContext.dstTensorDesc, 1, count, cdf);

            if(code != CUDNN_STATUS_SUCCESS){
                //If CuDNN can't infer algorithm - try IMPLICIT_GEMM
                //Why this specifically? According to the docs, it seems to have the least number of restrictions
                // to things like dilation

                OneTimeLogger.warn(log, "Error getting CuDNN forward algorithm - falling back on IMPLICIT_GEMM");
                mode = AlgoMode.USER_SPECIFIED;
                fwdAlgo = FwdAlgo.IMPLICIT_GEMM;
                algo[0] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            }

            algo[0] = cdf.algo();
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

        code = cudnnSetStream(cudnnContext, new CUstream_st(context.getCublasStream()));
        checkCudnn(true, "cudnnSetStream", code, input, weights, bias, null, kernel, strides, pad, mode, fwdAlgo, null, null, convolutionMode, dilation);

        code = cudnnGetConvolutionForwardWorkspaceSize(cudnnContext, cudnnContext.srcTensorDesc,
                cudnnContext.filterDesc, cudnnContext.convDesc, cudnnContext.dstTensorDesc, algo[0],
                sizeInBytes);
        checkCudnn(true, "cudnnGetConvolutionForwardWorkspaceSize", code, input, weights, bias, null, kernel, strides, pad, mode, fwdAlgo, null, null, convolutionMode, dilation);

        DataCache workSpace = workspaceMgr.getHelperWorkspace(LayerWorkspaceMgr.CUDNN_WORKSPACE_KEY);
        if (workSpace == null || sizeInBytes.get(0) > workSpace.capacity()) {
            if(log.isTraceEnabled()){
                if(workSpace == null){
                    log.trace("CudnnConvolutionHelper preOutput: allocating initial workspace of size {} ({})",
                            sizeInBytes.get(), BinaryByteUnit.format(sizeInBytes.get(), "#.00"));
                } else {
                    log.trace("CudnnConvolutionHelper preOutput: Deallocating workspace of size {} ({}), allocating new workspace of size {} ({})",
                            workSpace.capacity(), BinaryByteUnit.format(workSpace.capacity(), "#.00"),
                            sizeInBytes.get(), BinaryByteUnit.format(sizeInBytes.get(), "#.00"));
                }
            }
            if(workSpace != null)
                workSpace.deallocate();
            workSpace = new DataCache(sizeInBytes.get(0));
            workspaceMgr.setHelperWorkspace(LayerWorkspaceMgr.CUDNN_WORKSPACE_KEY, workSpace);
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

        if(origNHWC){
            z = z.permute(0,2,3,1);     //NCHW to NHWC
        }

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
    public INDArray activate(INDArray z, IActivation afn, boolean training) {
        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        INDArray activation = z;

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareAction(z);
        Pointer dstData = allocator.getPointer(z, context);

        checkCudnn(cudnnSetStream(cudnnContext, new CUstream_st(context.getCublasStream())));
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

    /**
     * @param poolingType     Used when preparing data for subsampling layers ONLY. Null for convolution layers
     * @return
     */
    public static CudnnForwardArgs getCudnnForwardArgs(INDArray input, int[] kernel, int[] strides, int[] padding, int[] dilation,
                                                       ConvolutionMode convolutionMode, PoolingType poolingType, CNN2DFormat format){
        INDArray origInput = input;

        //Check if we need to dup the input: views, non-contiguous, etc. CuDNN also seems to have has issues if strides
        // are non-default for C order - even if they *should* be OK otherwise
        if(input.isView() || !Shape.hasDefaultStridesForShape(input)){
            input = input.dup('c');
        }

        boolean nchw = format == CNN2DFormat.NCHW;
        int hIdx = nchw ? 2 : 1;
        int wIdx = nchw ? 3 : 2;

        val inH = input.size(hIdx);
        val inW = input.size(wIdx);

        boolean manualPadBottom = false;
        boolean manualPadRight = false;

        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, null, convolutionMode, dilation, format); //Also performs validation
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
                long[] newShape;
                if(nchw){
                    newShape = new long[]{input.size(0), input.size(1),
                            input.size(2) + (manualPadBottom ? 1 : 0),
                            input.size(3) + (manualPadRight ? 1 : 0)};
                } else {
                    newShape = new long[]{input.size(0),
                            input.size(1) + (manualPadBottom ? 1 : 0),
                            input.size(2) + (manualPadRight ? 1 : 0),
                            input.size(3)};
                }
                INDArray newInput;
                if(poolingType == null || poolingType != PoolingType.MAX){
                    newInput = Nd4j.create(input.dataType(), newShape);
                } else {
                    //For max pooling, we don't want to include the padding in the maximum values. But, CuDNN doesn't knowm
                    // that these values are padding and hence should be excluded. Instead: We'll use -infinity so that,
                    // if the 'real' (non-padding) values are all < 0, we take the real value, not the padding value
                    newInput = Nd4j.valueArrayOf(newShape, Double.NEGATIVE_INFINITY, input.dataType());
                }

                if(nchw){
                    newInput.put(new INDArrayIndex[]{all(), all(), interval(0,input.size(2)),
                            interval(0, input.size(3))}, input);
                } else {
                    newInput.put(new INDArrayIndex[]{all(), interval(0,input.size(1)),
                            interval(0, input.size(2)), all()}, input);
                }

                input = newInput;
                //Now: we've manually applied the "extra" bottom/right padding only - if required. Consequently, we
                // now have the same amount of padding required for top/bottom, and left/right - which we'll let
                // CuDNN handle
            }
        } else {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, padding, convolutionMode, dilation, format); //Also performs validation
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

    @Override
    public Map<String, Long> helperMemoryUse() {
        //No memory use other than shared, and the structs (which are small)
        return Collections.emptyMap();
    }

}