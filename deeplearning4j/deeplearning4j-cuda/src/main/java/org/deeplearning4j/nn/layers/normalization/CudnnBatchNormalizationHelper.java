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

package org.deeplearning4j.nn.layers.normalization;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseCudnnHelper;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.HashMap;
import java.util.Map;

import org.bytedeco.cuda.cudart.*;
import org.bytedeco.cuda.cudnn.*;

import static org.bytedeco.cuda.global.cudnn.*;

/**
 * cuDNN-based helper for the batch normalization layer.
 *
 * @author saudet
 */
@Slf4j
public class CudnnBatchNormalizationHelper extends BaseCudnnHelper implements BatchNormalizationHelper {

    public CudnnBatchNormalizationHelper(DataType dataType) {
        super(dataType);
    }

    private static class CudnnBatchNormalizationContext extends CudnnContext {

        private static class Deallocator extends CudnnBatchNormalizationContext implements Pointer.Deallocator {
            Deallocator(CudnnBatchNormalizationContext c) {
                super(c);
            }

            @Override
            public void deallocate() {
                destroyHandles();
            }
        }

        private cudnnTensorStruct srcTensorDesc = new cudnnTensorStruct(), dstTensorDesc = new cudnnTensorStruct(),
                        deltaTensorDesc = new cudnnTensorStruct(), gammaBetaTensorDesc = new cudnnTensorStruct();

        public CudnnBatchNormalizationContext() {
            createHandles();
            deallocator(new Deallocator(this));
        }

        public CudnnBatchNormalizationContext(CudnnBatchNormalizationContext c) {
            super(c);
            srcTensorDesc = new cudnnTensorStruct(c.srcTensorDesc);
            dstTensorDesc = new cudnnTensorStruct(c.dstTensorDesc);
            deltaTensorDesc = new cudnnTensorStruct(c.deltaTensorDesc);
            gammaBetaTensorDesc = new cudnnTensorStruct(c.gammaBetaTensorDesc);
        }

        @Override
        protected void createHandles() {
            super.createHandles();
            checkCudnn(cudnnCreateTensorDescriptor(srcTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(dstTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(deltaTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(gammaBetaTensorDesc));
        }

        @Override
        protected void destroyHandles() {
            checkCudnn(cudnnDestroyTensorDescriptor(srcTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(dstTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(deltaTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(gammaBetaTensorDesc));
            super.destroyHandles();
        }
    }

    protected final int batchNormMode = CUDNN_BATCHNORM_SPATIAL; // would need to increase rank of gamma and beta for CUDNN_BATCHNORM_PER_ACTIVATION

    private CudnnBatchNormalizationContext cudnnContext = new CudnnBatchNormalizationContext();
    private INDArray meanCache;
    private INDArray varCache;
    private double eps;

    public boolean checkSupported(double eps, boolean isFixedGammaBeta) {
        boolean supported = checkSupported();
        if (eps < CUDNN_BN_MIN_EPSILON) {
            supported = false;
            log.warn("Not supported: eps < CUDNN_BN_MIN_EPSILON (" + eps + " < " + CUDNN_BN_MIN_EPSILON + ")");
        }
        return supported;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray epsilon, long[] shape, INDArray gamma, INDArray beta,
                    INDArray dGammaView, INDArray dBetaView, double eps, LayerWorkspaceMgr layerWorkspaceMgr) {
        this.eps = eps;
        val miniBatch = (int) input.size(0);
        val depth = (int) input.size(1);
        val inH = (int) input.size(2);
        val inW = (int) input.size(3);

        final boolean isHalf = (input.dataType() == DataType.HALF);
        INDArray gammaOrig = null;
        INDArray dGammaViewOrig = null;
        INDArray dBetaViewOrig = null;
        if(isHalf) {    //Convert FP16 to FP32 if required (CuDNN BN doesn't support FP16 for these params, only for input/output)
            gammaOrig = gamma;
            dGammaViewOrig = dGammaView;
            dBetaViewOrig = dBetaView;
            /*
            From CuDNN docs: bnScale, resultBnScaleDiff, resultBnBiasDiff, savedMean, savedInvVariance
            "Note: The data type of this tensor descriptor must be 'float' for FP16 and FP32 input tensors, and 'double'
            for FP64 input tensors."
            >> Last 2 are the meanCache and varCache; first 3 are below
             */
            gamma = gamma.castTo(DataType.FLOAT);
            dGammaView = dGammaView.castTo(DataType.FLOAT);
            dBetaView = dBetaView.castTo(DataType.FLOAT);
        }

        Gradient retGradient = new DefaultGradient();

        if (!Shape.hasDefaultStridesForShape(epsilon)) {
            // apparently not supported by cuDNN
            epsilon = epsilon.dup('c');
        }

        val srcStride = ArrayUtil.toInts(input.stride());
        val deltaStride = ArrayUtil.toInts(epsilon.stride());

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.srcTensorDesc, dataType, (int) miniBatch, (int) depth, (int) inH, (int) inW,
                (int) srcStride[0], (int) srcStride[1], (int) srcStride[2], (int) srcStride[3]));
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.deltaTensorDesc, dataType, (int) miniBatch, (int) depth, (int) inH, (int) inW,
                (int) deltaStride[0], (int) deltaStride[1], (int) deltaStride[2], (int) deltaStride[3]));

        INDArray nextEpsilon = layerWorkspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, input.dataType(), new long[] {miniBatch, depth, inH, inW}, 'c');
        val dstStride = ArrayUtil.toInts(nextEpsilon.stride());

        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.dstTensorDesc, dataType, miniBatch, depth, inH, inW,
                        dstStride[0], dstStride[1], dstStride[2], dstStride[3]));
        checkCudnn(cudnnSetTensor4dDescriptor(cudnnContext.gammaBetaTensorDesc, TENSOR_FORMAT, toCudnnDataType(gamma.data().dataType()), (int)shape[0],
                (int)shape[1], shape.length > 2 ? (int)shape[2] : 1, shape.length > 3 ? (int)shape[3] : 1));

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareActionAllWrite(input, epsilon, nextEpsilon, gamma,
                        dGammaView, dBetaView);
        Pointer srcData = allocator.getPointer(input, context);
        Pointer epsData = allocator.getPointer(epsilon, context);
        Pointer dstData = allocator.getPointer(nextEpsilon, context);
        Pointer gammaData = allocator.getPointer(gamma, context);
        Pointer dGammaData = allocator.getPointer(dGammaView, context);
        Pointer dBetaData = allocator.getPointer(dBetaView, context);
        Pointer meanCacheData = allocator.getPointer(meanCache, context);
        Pointer varCacheData = allocator.getPointer(varCache, context);

        checkCudnn(cudnnSetStream(cudnnContext, new CUstream_st(context.getCublasStream())));
        checkCudnn(cudnnBatchNormalizationBackward(cudnnContext, batchNormMode, alpha, this.beta, alpha, alpha,
                        cudnnContext.srcTensorDesc, srcData, cudnnContext.deltaTensorDesc, epsData,
                        cudnnContext.dstTensorDesc, dstData, cudnnContext.gammaBetaTensorDesc, gammaData, dGammaData,
                        dBetaData, eps, meanCacheData, varCacheData));

        allocator.getFlowController().registerActionAllWrite(context, input, epsilon, nextEpsilon, gamma, dGammaView,
                        dBetaView);

        retGradient.setGradientFor(BatchNormalizationParamInitializer.GAMMA, dGammaView);
        retGradient.setGradientFor(BatchNormalizationParamInitializer.BETA, dBetaView);

        context.syncOldStream();

        //Convert back and assign, if required:
        if(isHalf){
            gammaOrig.assign(gamma.castTo(DataType.HALF));
            dGammaViewOrig.assign(dGammaView.castTo(DataType.HALF));
            dBetaViewOrig.assign(dBetaView.castTo(DataType.HALF));
        }

        return new Pair<>(retGradient, nextEpsilon);
    }


    @Override
    public INDArray preOutput(INDArray x, boolean training, long[] shape, INDArray gamma, INDArray beta, INDArray mean,
                    INDArray var, double decay, double eps, LayerWorkspaceMgr workspaceMgr) {
        this.eps = eps;
        final boolean isHalf = (x.dataType() == DataType.HALF);
        INDArray origGamma = gamma;
        INDArray origBeta = beta;
        INDArray origMean = mean;
        INDArray origVar = var;
        if(isHalf) {
            gamma = gamma.castTo(DataType.FLOAT);
            beta = beta.castTo(DataType.FLOAT);
            mean = mean.castTo(DataType.FLOAT);
            var = var.castTo(DataType.FLOAT);
        }

        //Notation difference between CuDNN and our implementation:
        //Us:       runningMean = (1-decay) * batchMean + decay * runningMean
        //CuDNN:    runningMean = decay * batchMean + (1-decay) * runningMean
        //i.e., "decay" has a different meaning...
        //Disable in-place updating of running mean/variance, so that all parameter changes are done via the update/gradient
        // vector. This is necessary for BatchNormalization to be safe to use in distributed gradient sharing settings
        decay = 0.0;                //From cudnn docs: runningMean = newMean*factor + runningMean*(1-factor). -> 0 = "in-place modification of running mean disabled"

        val miniBatch = (int) x.size(0);
        val inDepth = (int) x.size(1);
        val inH = (int) x.size(2);
        val inW = (int) x.size(3);

        val srcStride = ArrayUtil.toInts(x.stride());
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.srcTensorDesc, dataType, miniBatch, inDepth, inH, inW,
                        srcStride[0], srcStride[1], srcStride[2], srcStride[3]));

        INDArray activations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, x.dataType(), new long[] {miniBatch, inDepth, inH, inW}, 'c');

        val dstStride = ArrayUtil.toInts(activations.stride());
        checkCudnn(cudnnSetTensor4dDescriptorEx(cudnnContext.dstTensorDesc, dataType, miniBatch, inDepth, inH, inW,
                        dstStride[0], dstStride[1], dstStride[2], dstStride[3]));

        checkCudnn(cudnnSetTensor4dDescriptor(cudnnContext.gammaBetaTensorDesc, TENSOR_FORMAT, toCudnnDataType(mean.data().dataType()), (int)shape[0],
                (int)shape[1], shape.length > 2 ? (int)shape[2] : 1, shape.length > 3 ? (int)shape[3] : 1));

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context =
                        allocator.getFlowController().prepareActionAllWrite(x, activations, gamma, beta, mean, var);
        Pointer srcData = allocator.getPointer(x, context);
        Pointer dstData = allocator.getPointer(activations, context);
        Pointer gammaData = allocator.getPointer(gamma, context);
        Pointer betaData = allocator.getPointer(beta, context);
        Pointer meanData = allocator.getPointer(mean, context);
        Pointer varData = allocator.getPointer(var, context);

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        checkCudnn(cudnnSetStream(cudnnContext, new CUstream_st(context.getCublasStream())));
        if (training) {
            if(meanCache == null || meanCache.length() < mean.length()){
                try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    meanCache = Nd4j.createUninitialized(x.dataType(), mean.length());
                }
                if(x.dataType() == DataType.HALF){
                    try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                        meanCache = meanCache.castTo(DataType.FLOAT);
                    }
                }
            }
            if(varCache == null || varCache.length() < mean.length()){
                try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    varCache = Nd4j.createUninitialized(x.dataType(), mean.length());
                }
                if(nd4jDataType == DataType.HALF){
                    try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                        varCache = varCache.castTo(DataType.FLOAT);
                    }
                }
            }
            Pointer meanCacheData = allocator.getPointer(meanCache, context);
            Pointer varCacheData = allocator.getPointer(varCache, context);

            checkCudnn(cudnnBatchNormalizationForwardTraining(cudnnContext, batchNormMode, this.alpha, this.beta,
                            cudnnContext.srcTensorDesc, srcData, cudnnContext.dstTensorDesc, dstData,
                            cudnnContext.gammaBetaTensorDesc, gammaData, betaData, decay, meanData, varData, eps,
                            meanCacheData, varCacheData));
        } else {
            checkCudnn(cudnnBatchNormalizationForwardInference(cudnnContext, batchNormMode, this.alpha, this.beta,
                            cudnnContext.srcTensorDesc, srcData, cudnnContext.dstTensorDesc, dstData,
                            cudnnContext.gammaBetaTensorDesc, gammaData, betaData, meanData, varData, eps));
        }

        allocator.getFlowController().registerActionAllWrite(context, x, activations, gamma, beta, mean, var);

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            context.syncOldStream();

        context.syncOldStream();
        if(training) {
            AtomicAllocator.getInstance().getAllocationPoint(meanCache).tickDeviceWrite();
            AtomicAllocator.getInstance().getAllocationPoint(varCache).tickDeviceWrite();
        }

        if(training && isHalf){
            //Update the running mean and variance arrays; also gamma/beta
            origMean.assign(mean.castTo(DataType.HALF));
            origVar.assign(var.castTo(DataType.HALF));
            origGamma.assign(gamma.castTo(DataType.HALF));
            origBeta.assign(beta.castTo(DataType.HALF));
        }

        return activations;
    }

    @Override
    public INDArray getMeanCache(DataType dataType) {
        if(dataType == DataType.HALF){
            //Buffer is FP32
            return meanCache.castTo(DataType.HALF);
        }
        return meanCache;
    }

    @Override
    public INDArray getVarCache(DataType dataType) {
        INDArray ret;
        if(dataType == DataType.HALF){
            INDArray vc = varCache.castTo(DataType.HALF);
            ret = vc.mul(vc).rdivi(1.0).subi(eps);
        } else {
            ret = varCache.mul(varCache).rdivi(1.0).subi(eps);
        }
        if(dataType == DataType.HALF){
            //Buffer is FP32
            return ret.castTo(DataType.HALF);
        }
        return ret;
    }


    @Override
    public Map<String, Long> helperMemoryUse() {
        Map<String,Long> memUse = new HashMap<>();
        memUse.put("meanCache", meanCache == null ? 0 : meanCache.length() * meanCache.data().getElementSize());
        memUse.put("varCache", varCache == null ? 0 : varCache.length() * varCache.data().getElementSize());
        return memUse;
    }
}
