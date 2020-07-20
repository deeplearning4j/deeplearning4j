/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.cuda.convolution.subsampling;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.cuda.BaseCudnnHelper;
import org.deeplearning4j.cuda.convolution.CudnnConvolutionHelper;
import org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingHelper;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import java.util.Collections;
import java.util.Map;

import org.bytedeco.cuda.cudart.*;
import org.bytedeco.cuda.cudnn.*;

import static org.bytedeco.cuda.global.cudnn.*;
import static org.deeplearning4j.cuda.convolution.CudnnConvolutionHelper.getCudnnForwardArgs;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/**
 * cuDNN-based helper for the subsampling layer.
 *
 * @author saudet
 */
@Slf4j
public class CudnnSubsamplingHelper extends BaseCudnnHelper implements SubsamplingHelper {

    public CudnnSubsamplingHelper(DataType dataType) {
        super(dataType);
    }

    private static class CudnnSubsamplingContext extends CudnnContext {

        private static class Deallocator extends CudnnSubsamplingContext implements Pointer.Deallocator {
            Deallocator(CudnnSubsamplingContext c) {
                super(c);
            }

            @Override
            public void deallocate() {
                destroyHandles();
            }
        }

        private cudnnTensorStruct srcTensorDesc = new cudnnTensorStruct(), dstTensorDesc = new cudnnTensorStruct(),
                        deltaTensorDesc = new cudnnTensorStruct();
        private cudnnPoolingStruct poolingDesc = new cudnnPoolingStruct();

        public CudnnSubsamplingContext() {
            createHandles();
            deallocator(new Deallocator(this));
        }

        public CudnnSubsamplingContext(CudnnSubsamplingContext c) {
            super(c);
            srcTensorDesc = new cudnnTensorStruct(c.srcTensorDesc);
            dstTensorDesc = new cudnnTensorStruct(c.dstTensorDesc);
            deltaTensorDesc = new cudnnTensorStruct(c.deltaTensorDesc);
            poolingDesc = new cudnnPoolingStruct(c.poolingDesc);
        }

        @Override
        protected void createHandles() {
            super.createHandles();
            checkCudnn(cudnnCreateTensorDescriptor(srcTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(dstTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(deltaTensorDesc));
            checkCudnn(cudnnCreatePoolingDescriptor(poolingDesc));
        }

        @Override
        protected void destroyHandles() {
            checkCudnn(cudnnDestroyPoolingDescriptor(poolingDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(srcTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(dstTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(deltaTensorDesc));
            super.destroyHandles();
        }
    }

    private CudnnSubsamplingContext cudnnContext = new CudnnSubsamplingContext();

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray epsilon, int[] kernel, int[] strides,
                                                     int[] pad, PoolingType poolingType, ConvolutionMode convolutionMode,
                                                     int[] dilation, CNN2DFormat format, LayerWorkspaceMgr workspaceMgr) {
        if(dilation[0] != 1 || dilation[1] != 1){
            //CuDNN doesn't support dilated subsampling
            return null;
        }

        boolean nchw = format == CNN2DFormat.NCHW;
        int chIdx = nchw ? 1 : 3;
        int hIdx = nchw ? 2 : 1;
        int wIdx = nchw ? 3 : 2;

        //We require the output as one of the arguments for backprop here
        //TODO we could add cache mode support here somehow...
        INDArray reduced = activate(input, true, kernel, strides, pad, poolingType, convolutionMode, dilation, format, workspaceMgr);

        val miniBatch = input.size(0);
        val depth = input.size(chIdx);

        CudnnConvolutionHelper.CudnnForwardArgs args = getCudnnForwardArgs(input, kernel, strides, pad, dilation, convolutionMode, poolingType, format);
        input = args.getInput();
        val inH = input.size(hIdx);
        val inW = input.size(wIdx);
        val srcStride = input.stride();
        int[] outSize = args.getOutSize();
        int outH = outSize[0];
        int outW = outSize[1];

        //subsampling doesn't have weights and thus gradients are not calculated for this layer
        //only scale and reshape epsilon
        Gradient retGradient = new DefaultGradient();

        //Epsilons in shape: [miniBatch, channels, outH, outW]
        //Epsilons out shape: [miniBatch, channels, inH, inW]

        int poolingMode;
        switch (poolingType) {
            case AVG:
                poolingMode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
                break;
            case MAX:
                poolingMode = CUDNN_POOLING_MAX;
                break;
            default:
                return null;
        }

        if (!Shape.hasDefaultStridesForShape(epsilon) || epsilon.isView()) {
            // apparently not supported by cuDNN
            epsilon = epsilon.dup('c');
        }

        input = input.dup();

        val deltaStride = epsilon.stride();

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.srcTensorDesc, dataType, (int) miniBatch, (int) depth, (int) inH, (int) inW,
                (int) srcStride[0], (int) srcStride[chIdx], (int) srcStride[hIdx], (int) srcStride[wIdx]));
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.deltaTensorDesc, dataType, (int) miniBatch, (int) depth, (int) outH, (int) outW,
                (int) deltaStride[0], (int) deltaStride[chIdx], (int) deltaStride[hIdx], (int) deltaStride[wIdx]));
        checkCudnn(cudnnSetPooling2dDescriptor(cudnnContext.poolingDesc, poolingMode, CUDNN_PROPAGATE_NAN, kernel[0],
                        kernel[1], pad[0], pad[1], strides[0], strides[1]));

        long[] outEpsShape = nchw ? new long[] {miniBatch, depth, inH, inW} : new long[] {miniBatch, inH, inW, depth};
        INDArray outEpsilon = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, input.dataType(), outEpsShape, 'c');

        val dstStride = outEpsilon.stride();
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.dstTensorDesc, dataType, (int) miniBatch, (int) depth, (int) inH, (int) inW,
                (int) dstStride[0], (int) dstStride[chIdx], (int) dstStride[hIdx], (int) dstStride[wIdx]));

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareAction(input, epsilon, reduced, outEpsilon);
        Pointer srcData = allocator.getPointer(input, context);
        Pointer epsData = allocator.getPointer(epsilon, context);
        Pointer zData = allocator.getPointer(reduced, context);
        Pointer dstData = allocator.getPointer(outEpsilon, context);

        checkCudnn(cudnnSetStream(cudnnContext, new CUstream_st(context.getCublasStream())));
        checkCudnn(cudnnPoolingBackward(cudnnContext, cudnnContext.poolingDesc, alpha, cudnnContext.deltaTensorDesc,
                        zData, cudnnContext.deltaTensorDesc, epsData, cudnnContext.srcTensorDesc, srcData, beta,
                        cudnnContext.dstTensorDesc, dstData));

        allocator.registerAction(context, outEpsilon, input, epsilon, reduced);

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            context.syncOldStream();

        //Note that: if we had to manually pad for SAME mode, we have to 'undo' this manual padding for the epsilon
        // we return. The returned epsilon (i.e., dL/dIn array) has to be the same shape as the *original* input.
        if(args.isManualPadBottom() || args.isManualPadRight()) {
            if(nchw){
                outEpsilon = outEpsilon.get(all(), all(),
                        interval(0, outEpsilon.size(2) - (args.isManualPadBottom() ? 1 : 0)),
                        interval(0, outEpsilon.size(3) - (args.isManualPadRight() ? 1 : 0)));
            } else {
                outEpsilon = outEpsilon.get(all(),
                        interval(0, outEpsilon.size(1) - (args.isManualPadBottom() ? 1 : 0)),
                        interval(0, outEpsilon.size(2) - (args.isManualPadRight() ? 1 : 0)),
                        all());
            }
        }

        return new Pair<>(retGradient, outEpsilon);
    }


    @Override
    public INDArray activate(INDArray input, boolean training, int[] kernel, int[] strides, int[] pad,
                    PoolingType poolingType, ConvolutionMode convolutionMode, int[] dilation, CNN2DFormat format, LayerWorkspaceMgr workspaceMgr) {
        if(dilation[0] != 1 || dilation[1] != 1){
            //CuDNN doesn't support dilated subsampling
            return null;
        }

        boolean nchw = format == CNN2DFormat.NCHW;
        int chIdx = nchw ? 1 : 3;
        int hIdx = nchw ? 2 : 1;
        int wIdx = nchw ? 3 : 2;

        val miniBatch = input.size(0);
        val inDepth = input.size(nchw ? 1 : 3);

        CudnnConvolutionHelper.CudnnForwardArgs args = getCudnnForwardArgs(input, kernel, strides, pad, dilation, convolutionMode, poolingType, format);
        input = args.getInput();
        val inH = input.size(nchw ? 2 : 1);
        val inW = input.size(nchw ? 3 : 2);
        val srcStride = input.stride();
        val outSize = args.getOutSize();
        int outH = outSize[0];
        int outW = outSize[1];


        int poolingMode;
        switch (poolingType) {
            case AVG:
                poolingMode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
                break;
            case MAX:
                poolingMode = CUDNN_POOLING_MAX;
                break;
            default:
                return null;
        }

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        checkCudnn(cudnnSetPooling2dDescriptor(cudnnContext.poolingDesc, poolingMode, CUDNN_PROPAGATE_NAN, kernel[0],
                        kernel[1], pad[0], pad[1], strides[0], strides[1]));
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.srcTensorDesc, dataType, (int) miniBatch, (int) inDepth, (int) inH, (int) inW,
                (int) srcStride[0], (int) srcStride[chIdx], (int) srcStride[hIdx], (int) srcStride[wIdx]));

        long[] outShape = nchw ? new long[] {miniBatch, inDepth, outH, outW} : new long[] {miniBatch, outH, outW, inDepth};
        INDArray reduced = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, input.dataType(), outShape, 'c');

        val dstStride = reduced.stride();
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.dstTensorDesc, dataType, (int) miniBatch, (int) inDepth, (int) outH, (int) outW,
                (int) dstStride[0], (int) dstStride[chIdx], (int) dstStride[hIdx], (int) dstStride[wIdx]));

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareAction(input, reduced);
        Pointer srcData = allocator.getPointer(input, context);
        Pointer dstData = allocator.getPointer(reduced, context);

        checkCudnn(cudnnSetStream(cudnnContext, new CUstream_st(context.getCublasStream())));
        checkCudnn(cudnnPoolingForward(cudnnContext, cudnnContext.poolingDesc, alpha, cudnnContext.srcTensorDesc,
                        srcData, beta, cudnnContext.dstTensorDesc, dstData));

        allocator.registerAction(context, reduced, input);

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            context.syncOldStream();

        return reduced;
    }

    @Override
    public Map<String, Long> helperMemoryUse() {
        //No persistent memory use other than the structs (which are small)
        return Collections.emptyMap();
    }

}
